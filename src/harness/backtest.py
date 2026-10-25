"""Vectorised backtest. The one rule: a signal decided at bar t's close can only earn bar t+1's return."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Costs:
    fee_bps: float = 10.0       # exchange fee per unit of turnover (Coinbase Advanced taker is higher for small accounts)
    slippage_bps: float = 5.0   # half-spread plus market impact

    @property
    def per_turnover(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 1e4


def run(prices: pd.Series, signal: pd.Series, costs: Costs = Costs()) -> pd.DataFrame:
    """signal[t] is the target position (-1..1) chosen at the close of bar t using data up to t."""
    signal = signal.reindex(prices.index).clip(-1, 1).fillna(0.0)
    ret = prices.pct_change().fillna(0.0)
    pos = signal.shift(1).fillna(0.0)  # the shift(1): the position held through bar t was chosen at t-1
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * costs.per_turnover
    gross = pos * ret
    net = gross - cost
    return pd.DataFrame({
        "ret": ret, "pos": pos, "turnover": turnover, "cost": cost,
        "gross": gross, "net": net, "equity": (1 + net).cumprod(),
    })
