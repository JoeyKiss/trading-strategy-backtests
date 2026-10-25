"""Download and cache candles from Coinbase's public API (no key needed)."""
import json
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

API = "https://api.exchange.coinbase.com/products/{product}/candles"
GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}  # the only values Coinbase accepts
MAX_CANDLES = 300  # per request
CACHE_DIR = Path(__file__).resolve().parents[2] / "data"
COLUMNS = ["open", "high", "low", "close", "volume"]


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "backtest-harness"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.load(r)
    if isinstance(body, dict):  # errors come back as {"message": ...}
        raise RuntimeError(f"Coinbase error: {body.get('message', body)}")
    return body


def fetch_candles(product: str = "BTC-USD", start: str = "2016-01-01",
                  granularity: int = 86400, refresh: bool = False) -> pd.DataFrame:
    """OHLCV bars indexed by UTC open time, oldest first. Cached to data/ after the first download."""
    if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", product):
        raise ValueError(f"product must look like BTC-USD, got {product!r}")
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {sorted(GRANULARITIES)}")

    cache = CACHE_DIR / f"{product}_{granularity}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col="time", parse_dates=True)[COLUMNS]

    end = datetime.now(timezone.utc)
    t = pd.Timestamp(start, tz="UTC").to_pydatetime()
    step = timedelta(seconds=granularity * (MAX_CANDLES - 1))  # start and end are both inclusive
    rows = []
    while t < end:
        t2 = min(t + step, end)
        rows += _get(f"{API.format(product=product)}?granularity={granularity}"
                     f"&start={t.isoformat()}&end={t2.isoformat()}")
        t = t2
        time.sleep(0.15)  # public limit is ~10 requests/s
    if not rows:
        raise RuntimeError(f"no candles returned for {product}")

    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()[COLUMNS]
    df = df[df.index + pd.Timedelta(seconds=granularity) <= end]  # drop the still-forming bar
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(cache)
    return df
