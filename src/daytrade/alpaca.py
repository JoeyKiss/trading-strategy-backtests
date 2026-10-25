"""Alpaca market data (free plan, SIP = full-market feed): 5-minute bars back to 2016-01-04.

Same output as daytrade.data.load(): regular-session bars (09:30-16:00 New York), tz-naive New York
timestamps, columns open/high/low/close/volume, adjusted for splits and dividends (adjustment=all).

Each symbol's whole history is cached in ONE file and re-downloaded together: adjustment=all rescales
every earlier price whenever a new dividend is paid, so files downloaded at different times would not
line up (a fake gap would appear where they meet). Keys are read from the project's .env at runtime and
are never printed, logged or written anywhere else."""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
CACHE = ROOT / "data" / "intraday" / "alpaca"
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
FIRST_DAY = "2016-01-01"  # SIP history on the free plan starts 2016-01-04
FREE_PLAN_DELAY = pd.Timedelta(minutes=16)  # the latest 15 minutes of SIP data are withheld
MIN_INTERVAL_S = 0.35  # ~170 requests/minute, under the free plan's 200
NY = "America/New_York"
COLS = ["open", "high", "low", "close", "volume"]

_last_call = 0.0
_rate_lock = threading.Lock()  # downloads may run in parallel threads; they share one rate limit


def parse_env(raw: bytes) -> dict:
    """KEY=value lines from a .env file in UTF-8 (with or without BOM) or UTF-16, as editors save them."""
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw else raw.decode("utf-8-sig")
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m and not line.lstrip().startswith("#"):
            out[m.group(1)] = m.group(2).strip("\"'")
    return out


def _keys() -> tuple[str, str]:
    env = parse_env(ENV_FILE.read_bytes()) if ENV_FILE.exists() else {}
    key = os.environ.get("ALPACA_API_KEY") or env.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY") or env.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(f"ALPACA_API_KEY / ALPACA_SECRET_KEY not found in {ENV_FILE} (is it saved?)")
    return key, secret


def _get(params: dict, keys: tuple[str, str], retries: int = 5) -> dict:
    global _last_call
    url = f"{BARS_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": keys[0], "APCA-API-SECRET-KEY": keys[1],
                                               "Accept": "application/json"})
    for attempt in range(retries):
        with _rate_lock:
            wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:  # rate limited or server hiccup: back off and retry
                time.sleep(float(e.headers.get("Retry-After") or 5 * (attempt + 1)))
                continue
            raise RuntimeError(f"Alpaca HTTP {e.code} for {params.get('symbols')}: {e.read()[:200]!r}") from None
        except urllib.error.URLError:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Alpaca request failed {retries} times for {params.get('symbols')}")


def fetch_bars(symbol: str, start: str, end: pd.Timestamp, timeframe: str = "5Min",
               adjustment: str = "all", get=_get) -> pd.DataFrame:
    """All bars between start and end (UTC index), following next_page_token."""
    keys = _keys() if get is _get else ("", "")
    params = {"symbols": symbol, "timeframe": timeframe, "start": pd.Timestamp(start, tz="UTC").isoformat(),
              "end": end.tz_convert("UTC").isoformat(), "limit": 10000, "adjustment": adjustment,
              "feed": "sip", "sort": "asc"}
    rows = []
    while True:
        page = get(params, keys)
        rows += (page.get("bars") or {}).get(symbol, [])
        token = page.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    if not rows:
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz="UTC"))
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], utc=True)
    return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})[COLS].astype(float)


def early_closes(years) -> set:
    """NYSE 13:00 closes: July 3 and Dec 24 when they fall Mon-Thu, and the day after Thanksgiving.
    (Matches every session since 2016 whose SIP volume after 13:00 collapses to after-hours levels.)"""
    out = set()
    for y in years:
        for d in (pd.Timestamp(y, 7, 3), pd.Timestamp(y, 12, 24)):
            if d.dayofweek <= 3:
                out.add(d)
        nov1 = pd.Timestamp(y, 11, 1)
        out.add(nov1 + pd.Timedelta(days=(3 - nov1.dayofweek) % 7 + 21 + 1))  # 4th Thursday + 1
    return out


def _drop_after_early_close(df: pd.DataFrame) -> pd.DataFrame:
    """SIP bars include after-hours trades; on 13:00-close days everything from 13:00 on is after hours."""
    if df.empty:
        return df
    day = df.index.normalize()
    late = day.isin(list(early_closes(range(day[0].year, day[-1].year + 1)))) & (df.index.hour >= 13)
    return df[~late]


def regular_session(df_utc: pd.DataFrame) -> pd.DataFrame:
    """UTC bars -> tz-naive New York time, 09:30-16:00 only (13:00 on early-close days; DST handled by
    the tz conversion)."""
    df = df_utc.copy()
    df.index = df.index.tz_convert(NY).tz_localize(None)
    t = df.index.hour * 60 + df.index.minute
    return _drop_after_early_close(df[(t >= 9 * 60 + 30) & (t < 16 * 60)][COLS])


def _cache_path(symbol: str, kind: str) -> Path:
    if not re.fullmatch(r"[A-Z0-9.]{1,10}", symbol):
        raise ValueError(f"bad symbol {symbol!r}")
    return CACHE / f"{symbol}_{kind}.parquet"


def _fresh(path: Path, max_age_days: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400


def load(symbol: str, max_age_days: float = 7, start: str = FIRST_DAY) -> pd.DataFrame:
    """Regular-session 5-minute bars from 2016, re-downloading the whole history if the cache is older
    than max_age_days (pass float('inf') to never download)."""
    path = _cache_path(symbol, "5Min")
    if not _fresh(path, max_age_days):
        end = pd.Timestamp.now(tz="UTC") - FREE_PLAN_DELAY
        df = _drop_unfinished_session(regular_session(fetch_bars(symbol, start, end)))
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        _save_raw_factor(symbol, start, end)
    return _drop_unfinished_session(_drop_after_early_close(pd.read_parquet(path)))  # older caches kept them


def _drop_unfinished_session(df: pd.DataFrame) -> pd.DataFrame:
    """Never keep (or cache) today's session before it has closed: it would later pass as a short full day."""
    now = pd.Timestamp.now(tz=NY)
    if len(df) and df.index[-1].date() == now.date() and (now.hour, now.minute) < (16, 15):
        return df[df.index.normalize() < now.normalize().tz_localize(None)]
    return df


def _save_raw_factor(symbol: str, start: str, end: pd.Timestamp) -> None:
    """Per-day ratio of the price actually traded (raw) to the adjusted price, for the 1-cent cost floor."""
    raw = fetch_bars(symbol, start, end, "1Day", "raw")["close"]
    adj = fetch_bars(symbol, start, end, "1Day", "all")["close"]
    f = (raw / adj).dropna()
    f.index = f.index.tz_convert(NY).tz_localize(None).normalize()
    f.rename("raw_over_adjusted").to_frame().to_parquet(_cache_path(symbol, "rawfactor"))


def raw_factor(symbol: str) -> pd.Series:
    """Date -> raw/adjusted price ratio (1.0 when nothing has been adjusted since)."""
    return pd.read_parquet(_cache_path(symbol, "rawfactor"))["raw_over_adjusted"]


def parity(yahoo: pd.DataFrame, alpaca: pd.DataFrame) -> pd.DataFrame:
    """Per shared session: bar counts in each source and how well the closes agree. Alpaca is adjusted for
    later dividends and Yahoo's intraday bars are not, so the price RATIO may sit slightly below 1 but
    must be constant within a session; its spread is what shows whether the bars are the same."""
    days = sorted(set(yahoo.index.normalize()) & set(alpaca.index.normalize()))
    out = []
    for d in days:
        y = yahoo[yahoo.index.normalize() == d]["close"]
        a = alpaca[alpaca.index.normalize() == d]["close"]
        both = y.index.intersection(a.index)
        ratio = (a[both] / y[both]) if len(both) else pd.Series(dtype=float)
        # the session's first open is what gap strategies trade on; it isn't covered by the close ratio
        first_open = (alpaca.loc[a.index[0], "open"] / yahoo.loc[y.index[0], "open"] / ratio.median() - 1
                      if len(ratio) and "open" in yahoo and "open" in alpaca and a.index[0] == y.index[0] else float("nan"))
        out.append({"date": d, "yahoo_bars": len(y), "alpaca_bars": len(a), "shared_bars": len(both),
                    "median_ratio": float(ratio.median()) if len(ratio) else float("nan"),
                    "ratio_spread": float((ratio / ratio.median() - 1).abs().quantile(0.95)) if len(ratio) else float("nan"),
                    "first_open_diff": float(first_open)})
    return pd.DataFrame(out)
