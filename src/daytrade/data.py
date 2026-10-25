"""Intraday bars from Yahoo's public chart API. Yahoo only serves ~60 days of 5-minute bars and
~730 days of hourly bars, so every download is merged into a local cache and history grows over time."""
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .sim import Day

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{s}?interval={i}&range={r}"
MAX_RANGE = {"5m": "60d", "15m": "60d", "60m": "730d"}
CACHE = Path(__file__).resolve().parents[2] / "data" / "intraday"
NY = "America/New_York"
COLS = ["open", "high", "low", "close", "volume"]


def _download(symbol: str, interval: str) -> pd.DataFrame:
    url = CHART.format(s=symbol, i=interval, r=MAX_RANGE[interval])
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(NY).tz_localize(None)
    return pd.DataFrame({c: q[c] for c in COLS}, index=idx, dtype=float).dropna()


def load(symbol: str, interval: str, refresh: bool = True) -> pd.DataFrame:
    """Regular-session bars (09:30-16:00 New York time), oldest first, cache merged with fresh data."""
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,12}", symbol) or interval not in MAX_RANGE:
        raise ValueError(f"bad symbol/interval {symbol!r} {interval!r}")
    path = CACHE / f"{symbol}_{interval}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
    if refresh:
        fresh = _download(symbol, interval)
        df = fresh if df is None else pd.concat([df, fresh])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        time.sleep(0.15)
    if df is None:
        raise FileNotFoundError(f"no cached {interval} bars for {symbol}; run without --no-refresh first")
    t = df.index.hour * 60 + df.index.minute
    df = df[(t >= 9 * 60 + 30) & (t < 16 * 60)]
    now = pd.Timestamp.now(tz=NY)
    if len(df) and df.index[-1].date() == now.date() and (now.hour, now.minute) < (16, 15):
        df = df[df.index.normalize() < now.normalize().tz_localize(None)]  # today's session isn't over
    return df[COLS]


def to_days(df: pd.DataFrame, min_bars: int = 3) -> tuple[list[Day], list[float | None], list[pd.Timestamp]]:
    """Split into sessions. Returns (days, previous session's close for each, session dates)."""
    days, prev_closes, dates = [], [], []
    prev = None
    for date, g in df.groupby(df.index.normalize()):
        if len(g) >= min_bars:
            mins = ((g.index - date) / pd.Timedelta(minutes=1)).to_numpy() - 570  # minutes since 09:30
            days.append(Day(*(g[c].to_numpy() for c in COLS), mins=mins.astype(int)))
            prev_closes.append(prev)
            dates.append(date)
        # a broken session (data outage) has no usable close: the next session gets no previous close
        prev = float(g["close"].iloc[-1]) if len(g) >= min_bars else None
    return days, prev_closes, dates
