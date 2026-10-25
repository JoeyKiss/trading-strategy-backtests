"""Performance metrics, including Bailey & Lopez de Prado's probabilistic and deflated Sharpe ratios."""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 365  # crypto trades every day
EULER_GAMMA = 0.5772156649
_N = NormalDist()


def sharpe(r: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Annualised by default; pass periods=1 for the per-bar Sharpe the PSR/DSR formulas use."""
    sd = r.std(ddof=1)
    if not sd or np.isnan(sd):
        return 0.0
    return float(r.mean() / sd * math.sqrt(periods))


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def cagr(r: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    growth = float((1 + r).prod())
    years = len(r) / periods
    return growth ** (1 / years) - 1 if growth > 0 and years > 0 else -1.0


def summary(bt: pd.DataFrame, periods: int = PERIODS_PER_YEAR) -> dict:
    r = bt["net"]
    return {
        "sharpe": sharpe(r, periods),
        "cagr": cagr(r, periods),
        "max_drawdown": max_drawdown(r),
        "total_return": float((1 + r).prod() - 1),
        "turnover_per_year": float(bt["turnover"].mean() * periods),
        "cost_per_year": float(bt["cost"].mean() * periods),
        "exposure": float((bt["pos"] != 0).mean()),
        "bars": len(r),
    }


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-bar Sharpe > sr_benchmark), allowing for sample length, skew and fat tails."""
    n = len(r)
    sr = sharpe(r, 1)
    skew, kurt = float(r.skew()), float(r.kurt()) + 3  # pandas kurt() is excess kurtosis
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if n < 3 or denom <= 0 or np.isnan(denom):
        return float("nan")
    return _N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom))


def expected_max_sharpe(trial_sharpes: list[float]) -> float:
    """The best per-bar Sharpe you'd expect from N tries of pure noise with this spread of results."""
    n = len(trial_sharpes)
    if n < 2:
        return 0.0
    sd = float(np.std(trial_sharpes, ddof=1))
    return sd * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n)
                 + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n * math.e)))


def deflated_sharpe(r: pd.Series, trial_sharpes: list[float]) -> float:
    """PSR against the luck-adjusted hurdle. Needs >= 0.95 to call the result real."""
    return probabilistic_sharpe(r, expected_max_sharpe(trial_sharpes))
