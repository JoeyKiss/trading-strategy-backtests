"""Example strategies. Each takes OHLCV bars and returns the target position decided at each
bar's close, using only data up to that close. The backtester applies the shift(1)."""
import pandas as pd


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.Series:
    """Long while the fast moving average is above the slow one, flat otherwise."""
    c = df["close"]
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float)


def momentum(df: pd.DataFrame, lookback: int = 30) -> pd.Series:
    """Long while the trailing `lookback`-bar return is positive, flat otherwise."""
    return (df["close"].pct_change(lookback) > 0).astype(float)


def cheater(df: pd.DataFrame) -> pd.Series:
    """Deliberately broken: peeks at the next close. Exists to prove the look-ahead check works."""
    return (df["close"].shift(-1) > df["close"]).astype(float)


# name -> (function, parameter grid). Every grid point counts as a trial for the deflated Sharpe.
STRATEGIES = {
    "sma_cross": (sma_cross, {"fast": [10, 20, 50], "slow": [100, 150, 200]}),
    "momentum": (momentum, {"lookback": [7, 14, 30, 60, 90]}),
    "cheater": (cheater, {}),
}
