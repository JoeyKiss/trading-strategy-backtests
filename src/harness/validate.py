"""The gates a strategy must clear before anyone should believe its backtest."""
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from . import backtest, metrics
from .strategies import buy_and_hold

Strategy = Callable[..., pd.Series]

SUSPICIOUS_SHARPE = 2.0  # on daily crypto, above this is usually a bug, not a discovery
MIN_DSR = 0.95


def param_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*grid.values())]


def find_lookahead(strategy: Strategy, df: pd.DataFrame, params: dict,
                   n_checks: int = 25, warmup: int = 250) -> list[pd.Timestamp]:
    """Recompute the signal on data cut off at bar k. If signal[k] changes, it was using bars after k."""
    full = strategy(df, **params)
    cuts = np.unique(np.linspace(min(warmup, len(df) - 2), len(df) - 2, n_checks).astype(int))
    bad = []
    for k in cuts:
        a, b = strategy(df.iloc[:k + 1], **params).iloc[-1], full.iloc[k]
        if not ((pd.isna(a) and pd.isna(b)) or np.isclose(a, b)):
            bad.append(df.index[k])
    return bad


def walk_forward(strategy: Strategy, df: pd.DataFrame, grid: dict, costs: backtest.Costs,
                 train: int, test: int) -> tuple[pd.DataFrame, list[float], list]:
    """Pick params on each train window by net Sharpe, then trade them on the following test window only.
    Signals are computed once on the full history, which is only safe after find_lookahead passes."""
    prices = df["close"]
    configs = param_grid(grid)
    signals = [strategy(df, **p) for p in configs]
    nets = [backtest.run(prices, s, costs)["net"] for s in signals]

    oos_signal = pd.Series(np.nan, index=df.index)
    chosen = []
    for start in range(train, len(df), test):
        scores = [metrics.sharpe(n.iloc[start - train:start]) for n in nets]
        best = int(np.argmax(scores))
        oos_signal.iloc[start:start + test] = signals[best].iloc[start:start + test]
        chosen.append((df.index[start], configs[best]))

    # One backtest over the stitched signal, so switching params between folds pays its trading costs too.
    oos = backtest.run(prices, oos_signal.fillna(0.0), costs).iloc[train:]
    trial_sharpes = [metrics.sharpe(n.iloc[train:], 1) for n in nets]
    return oos, trial_sharpes, chosen


def evaluate(strategy: Strategy, df: pd.DataFrame, grid: dict, costs: backtest.Costs = backtest.Costs(),
             train: int = 730, test: int = 90) -> dict:
    """Run every gate and return a report. verdict is PASS only if all gates pass."""
    if len(df) < train + test:
        raise ValueError(f"need at least {train + test} bars, got {len(df)}")
    prices = df["close"]
    configs = param_grid(grid)

    in_sample = {tuple(p.items()): metrics.summary(backtest.run(prices, strategy(df, **p), costs)) for p in configs}
    best_is = max(in_sample.values(), key=lambda s: s["sharpe"])
    report = {"strategy": strategy.__name__, "configs_tried": len(configs),
              "in_sample_best": best_is, "warnings": []}
    if best_is["sharpe"] > SUSPICIOUS_SHARPE:
        report["warnings"].append(f"in-sample Sharpe {best_is['sharpe']:.2f} > {SUSPICIOUS_SHARPE}: suspect a leak")

    leaks = {str(p): bars for p in configs if (bars := find_lookahead(strategy, df, p))}
    if leaks:
        report.update(verdict="FAIL", gates={"no_lookahead": False}, lookahead=leaks)
        return report

    oos, trial_sharpes, chosen = walk_forward(strategy, df, grid, costs, train, test)
    oos_summary = metrics.summary(oos)
    dsr = metrics.deflated_sharpe(oos["net"], trial_sharpes)
    bench = metrics.summary(backtest.run(prices, buy_and_hold(df), costs).iloc[train:])
    gates = {
        "no_lookahead": True,
        "oos_sharpe_positive": oos_summary["sharpe"] > 0,
        "deflated_sharpe": bool(dsr >= MIN_DSR),
    }
    report.update(
        verdict="PASS" if all(gates.values()) else "FAIL", gates=gates,
        out_of_sample=oos_summary, deflated_sharpe=dsr, buy_and_hold=bench,
        oos_start=str(oos.index[0].date()), params_chosen=chosen,
    )
    return report
