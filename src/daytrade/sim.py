"""Intraday execution. A strategy looks at one session and returns trades. Rules it must follow:
a trade entering at bar e may use bars before e plus bar e's OPEN, nothing else (find_lookahead checks).
Fills: entries at the bar's open; stops/targets inside a bar (stop wins if both are touched); planned
exits at a later bar's open; anything still open is closed at the session's last close."""
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple

import numpy as np


class Day(NamedTuple):
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    mins: np.ndarray  # minutes since 09:30 at each bar's start


@dataclass(frozen=True)
class Trade:
    entry: int                  # bar index, filled at that bar's open
    side: int                   # +1 long, -1 short
    exit: int | None = None     # planned exit at this bar's open; None = hold until stopped or the close
    stop: float | None = None
    target: float | None = None


# (day, previous session close or None, **params) -> trades
Strategy = Callable[..., list[Trade]]


def positions_to_trades(pos: np.ndarray) -> list[Trade]:
    """pos[i] = position wanted after bar i's close (-1, 0, +1). Each change trades at the next open.
    The last bar's decision is ignored: the session is over."""
    trades: list[Trade] = []
    cur = 0
    for i, p in enumerate(pos[:-1]):
        p = int(p)
        if p != cur:
            if cur != 0:
                trades[-1] = replace(trades[-1], exit=i + 1)
            if p != 0:
                trades.append(Trade(entry=i + 1, side=p))
            cur = p
    return trades


def _stop_or_target(t: Trade, d: Day, j: int) -> float | None:
    """Exit price if bar j touches the stop (checked first) or target, else None."""
    o, h, l = d.open[j], d.high[j], d.low[j]
    if t.stop is not None and (l <= t.stop if t.side > 0 else h >= t.stop):
        gapped = o <= t.stop if t.side > 0 else o >= t.stop
        return o if gapped else t.stop
    if t.target is not None and (h >= t.target if t.side > 0 else l <= t.target):
        gapped = o >= t.target if t.side > 0 else o <= t.target
        return o if gapped else t.target
    return None


def execute(d: Day, trades: list[Trade]) -> list[dict]:
    n = len(d.open)
    fills, free_from = [], 0
    for t in trades:
        if t.entry < free_from or t.entry >= n or (t.exit is not None and t.exit <= t.entry):
            continue
        entry_px, exit_px, exit_i, how = d.open[t.entry], None, n - 1, "close"
        last_held = (t.exit - 1) if t.exit is not None and t.exit < n else n - 1
        for j in range(t.entry, last_held + 1):
            px = _stop_or_target(t, d, j)
            if px is not None:
                exit_px, exit_i, how = px, j, "stop/target"
                break
        if exit_px is None:
            if t.exit is not None and t.exit < n:
                exit_px, exit_i, how = d.open[t.exit], t.exit, "signal"
            else:
                exit_px = d.close[n - 1]
        fills.append({"entry": t.entry, "exit": exit_i, "side": t.side, "how": how,
                      "gross": t.side * (exit_px / entry_px - 1)})
        free_from = exit_i if how == "signal" else exit_i + 1
    return fills


def run_days(fn: Strategy, params: dict, days: list[Day], prev_closes: list, cost_bps,
             long_only: bool = False) -> np.ndarray:
    """Per-session results, shape (n_days, 3): gross return, net return, number of trades.
    cost_bps is per side: one number, or one per session (e.g. with a 1-cent spread floor)."""
    costs = np.broadcast_to(np.asarray(cost_bps, dtype=float), (len(days),))
    out = np.zeros((len(days), 3))
    for k, (d, pc) in enumerate(zip(days, prev_closes)):
        trades = fn(d, pc, **params)
        if long_only:
            trades = [t for t in trades if t.side > 0]
        fills = execute(d, trades)
        gross = sum(f["gross"] for f in fills)
        out[k] = gross, gross - len(fills) * 2 * costs[k] / 1e4, len(fills)
    return out


def _truncate(d: Day, k: int) -> Day:
    """The session as known at bar k's open: bars before k, plus bar k's open price only."""
    cut = [a[:k + 1].copy() for a in d]
    for arr in (cut[1], cut[2], cut[3]):
        arr[k] = cut[0][k]
    cut[4][k] = 0
    return Day(*cut)


def find_lookahead(fn: Strategy, params: dict, d: Day, prev_close, checks: int = 12) -> list[int]:
    """Bars k where the trades entering at or before k change once later bars are hidden."""
    full = fn(d, prev_close, **params)
    bad = []
    for k in np.unique(np.linspace(0, len(d.open) - 1, checks).astype(int)):
        part = fn(_truncate(d, int(k)), prev_close, **params)
        a = [(t.entry, t.side, t.stop, t.target) for t in full if t.entry <= k]
        b = [(t.entry, t.side, t.stop, t.target) for t in part if t.entry <= k]
        exits_a = [t.exit for t in full if t.entry <= k and t.exit is not None and t.exit <= k]
        exits_b = [t.exit for t in part if t.entry <= k and t.exit is not None and t.exit <= k]
        if a != b or exits_a != exits_b:
            bad.append(int(k))
    return bad
