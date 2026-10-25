"""Paper-trading strategies. Each gets closing prices (USD, adjusted) up to and including today plus
the set of tickers it currently holds, and returns target weights to trade at the next open."""
import math

import numpy as np
import pandas as pd

BENCHMARK = "VT"  # Vanguard Total World Stock ETF
UNIVERSE = [
    "NVDA", "MU", "AMD", "TSM", "ASML", "AVGO", "MSFT", "GOOGL", "AMZN", "META", "AAPL", "NFLX",
    "LLY", "ABBV", "NVO", "AZN", "UNH", "BRK-B", "JPM", "V", "MA", "COST", "PG", "PEP", "KO",
    "MCD", "HD", "XOM", "CVX", "NEE", "CAT", "GE", "SAP",
]


def rsi(close: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Wilder's RSI: 0 after straight losses, 100 after straight gains."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 * gain / (gain + loss)


def world_buy_hold(closes: pd.DataFrame, held: set) -> dict:
    """Benchmark: put everything in a world ETF and never touch it."""
    return {BENCHMARK: 1.0}


def universe_equal(closes: pd.DataFrame, held: set) -> dict:
    """Benchmark for the stock strategies: own every stock in the list equally. Any stock-picking
    strategy has to beat this, or its 'edge' is just the list it was given."""
    available = [t for t in UNIVERSE if t in closes and closes[t].notna().iloc[-1]]
    return {t: 1 / len(available) for t in available}


def world_trend(closes: pd.DataFrame, held: set) -> dict:
    """World ETF while it's above its 200-day average, cash otherwise. Aims to sidestep long crashes."""
    s = closes[BENCHMARK].dropna()
    if len(s) < 200:
        return {}
    return {BENCHMARK: 1.0} if s.iloc[-1] > s.iloc[-200:].mean() else {}


def momentum_top5(closes: pd.DataFrame, held: set, n: int = 5, keep_within: int = 10) -> dict:
    """Hold the 5 stocks with the best 12-month return (skipping the last month) per unit of risk.
    A holding is only replaced once it falls out of the top 10, which keeps trading down."""
    c = closes[UNIVERSE].dropna(axis=1, thresh=253)
    if len(c) < 253:
        return {}
    mom = c.iloc[-22] / c.iloc[-253] - 1
    vol = np.log(c).diff().iloc[-252:].std() * math.sqrt(252)
    ranked = list((mom / vol).dropna().sort_values(ascending=False).index)
    keep = [t for t in ranked[:keep_within] if t in held]
    picks = (keep + [t for t in ranked if t not in keep])[:n]
    return {t: 1 / n for t in picks}


def rsi2_dip(closes: pd.DataFrame, held: set, max_positions: int = 5, entry_rsi: float = 10.0) -> dict:
    """Daily trading: buy stocks in a long-term uptrend (above the 200-day average) right after a sharp
    2-day drop (RSI(2) < 10), and sell on the first close above the 5-day average. Holds for days."""
    c = closes[UNIVERSE]
    if len(c) < 200:
        return {}
    last, sma200, sma5 = c.iloc[-1], c.iloc[-200:].mean(), c.iloc[-5:].mean()
    r = rsi(c.iloc[-30:]).iloc[-1]
    stay = [t for t in held if t in c and last[t] <= sma5[t]]
    candidates = r[(last > sma200) & (r < entry_rsi)].drop(stay, errors="ignore").sort_values()
    picks = (stay + list(candidates.index))[:max_positions]
    return {t: 1 / max_positions for t in picks}


STRATEGIES = {
    "world_buy_hold": world_buy_hold,
    "universe_equal": universe_equal,
    "world_trend": world_trend,
    "momentum_top5": momentum_top5,
    "rsi2_dip": rsi2_dip,
}
BENCHMARK_STRATEGY = "world_buy_hold"
BENCHMARKS = {BENCHMARK_STRATEGY, "universe_equal"}  # baselines, not counted as strategy trials
TICKERS = [BENCHMARK] + UNIVERSE
