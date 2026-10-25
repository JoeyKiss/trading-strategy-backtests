"""The trading loop. Each day: overnight move, fill yesterday's orders at the open, open-to-close move,
then the strategy decides from data up to today's close. Orders always fill on the NEXT open."""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .market import Panels

# closes up to and including today, tickers currently held -> target weights (>= 0, sum <= 1)
Strategy = Callable[[pd.DataFrame, set], dict]

LOOKBACK_ROWS = 300  # strategies see at most this many days; enough for a 12-month lookback


@dataclass(frozen=True)
class Costs:
    fee_eur: float = 1.0          # flat fee per order, typical of EU online brokers
    slippage_bps: float = 10.0    # spread plus EUR/USD conversion on US stocks
    min_trade_frac: float = 0.01  # skip top-ups/trims under 1% of equity; entries and exits always trade
    cash_buffer: float = 0.005    # keep 0.5% in cash so fees never push cash negative


@dataclass
class Account:
    name: str
    capital: float
    cash: float
    positions: dict = field(default_factory=dict)  # ticker -> EUR market value
    pending: dict | None = None                     # target weights to execute at the next open
    last_date: str | None = None
    equity: list = field(default_factory=list)      # [[date, eur], ...]
    trades: list = field(default_factory=list)
    fees: float = 0.0

    @property
    def value(self) -> float:
        return self.cash + sum(self.positions.values())


def _validate(weights: dict) -> dict:
    if any(w < 0 for w in weights.values()) or sum(weights.values()) > 1 + 1e-9:
        raise ValueError(f"weights must be >= 0 and sum to <= 1, got {weights}")
    return {t: float(w) for t, w in weights.items() if w > 0}


def _mark(acct: Account, ratio: pd.Series) -> None:
    for t in acct.positions:
        r = ratio.get(t, np.nan)
        if np.isfinite(r):
            acct.positions[t] *= r


def _rebalance(acct: Account, date: str, prices: pd.Series, costs: Costs) -> None:
    equity = acct.value
    investable = equity * (1 - costs.cash_buffer)
    targets = acct.pending or {}
    # Sells first, so the trade log reads in a natural order.
    for t in sorted(set(targets) | set(acct.positions), key=lambda t: targets.get(t, 0) - acct.positions.get(t, 0) > 0):
        if not np.isfinite(prices.get(t, np.nan)):
            continue  # no price today: can't trade it
        cur, tgt = acct.positions.get(t, 0.0), targets.get(t, 0.0) * investable
        delta = tgt - cur
        is_entry_or_exit = cur == 0 or tgt == 0
        if abs(delta) < 0.01 or (not is_entry_or_exit and abs(delta) < costs.min_trade_frac * equity):
            continue
        cost = costs.fee_eur + abs(delta) * costs.slippage_bps / 1e4
        acct.cash -= delta + cost
        acct.fees += cost
        if tgt == 0:
            acct.positions.pop(t, None)
        else:
            acct.positions[t] = tgt
        acct.trades.append({"date": date, "ticker": t, "side": "BUY" if delta > 0 else "SELL",
                            "eur": round(abs(delta), 2), "price_eur": round(float(prices[t]), 4),
                            "cost": round(cost, 2)})
    acct.pending = None


def decide(acct: Account, strategy: Strategy, panels: Panels, i: int) -> None:
    window = panels.close_usd.iloc[max(0, i - LOOKBACK_ROWS + 1):i + 1]
    acct.pending = _validate(strategy(window, set(acct.positions)))
    acct.last_date = panels.dates[i].date().isoformat()


def open_account(name: str, capital: float, strategy: Strategy, panels: Panels, start: str) -> Account:
    """Start at the close of `start` (or the last trading day before it). The first orders fill next open."""
    i = int(panels.dates.searchsorted(pd.Timestamp(start), side="right")) - 1
    if i < 0:
        raise ValueError(f"no market data on or before {start}")
    acct = Account(name=name, capital=capital, cash=capital)
    acct.equity.append([panels.dates[i].date().isoformat(), round(capital, 2)])
    decide(acct, strategy, panels, i)
    return acct


def advance(acct: Account, strategy: Strategy, panels: Panels, costs: Costs = Costs()) -> int:
    """Process every completed trading day after acct.last_date. Returns the number of days processed."""
    dates = panels.dates
    start = int(dates.searchsorted(pd.Timestamp(acct.last_date), side="right"))
    if start == 0:
        raise ValueError(f"panels must include {acct.last_date} to continue from it")
    o, c = panels.open_eur, panels.close_eur
    for i in range(start, len(dates)):
        d, prev = dates[i], dates[i - 1]
        _mark(acct, o.loc[d] / c.loc[prev])
        if acct.pending is not None:
            _rebalance(acct, d.date().isoformat(), o.loc[d], costs)
        _mark(acct, c.loc[d] / o.loc[d])
        acct.equity.append([d.date().isoformat(), round(acct.value, 2)])
        decide(acct, strategy, panels, i)
    return len(dates) - start


def save(accounts: list[Account], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(a) for a in accounts], indent=1))
    tmp.replace(path)  # atomic, so a crash mid-write can't corrupt the ledger


def load(path: Path) -> list[Account]:
    return [Account(**a) for a in json.loads(path.read_text())]


def returns(acct: Account) -> pd.Series:
    eq = pd.Series({pd.Timestamp(d): v for d, v in acct.equity})
    return eq.pct_change().dropna()
