"""Daily bars from Yahoo Finance's public chart API, turned into EUR price panels."""
import json
import re
import time
import urllib.request
from dataclasses import dataclass

import pandas as pd

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1={p1}&period2={p2}&interval=1d"
FX = "EURUSD=X"  # USD per 1 EUR
NY = "America/New_York"
MARKET_CLOSED_AFTER = (16, 15)  # NY time; before this, today's bar is still forming


@dataclass
class Panels:
    """Wide frames (dates x tickers). Positions are valued in EUR using split- and dividend-adjusted
    prices, so splits never break the books and dividends count as return."""
    open_eur: pd.DataFrame
    close_eur: pd.DataFrame
    close_usd: pd.DataFrame  # what strategies see; FX moves shouldn't drive stock signals

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close_eur.index


def _fetch(ticker: str, start: str) -> pd.DataFrame:
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,12}", ticker):
        raise ValueError(f"bad ticker {ticker!r}")
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    url = CHART.format(t=ticker, p1=p1, p2=int(time.time()))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)["chart"]["result"][0]
    if res["meta"].get("dataGranularity") != "1d":  # Yahoo silently downgrades long ranges to monthly
        raise RuntimeError(f"{ticker}: expected daily bars, got {res['meta'].get('dataGranularity')}")
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    tz = res["meta"].get("exchangeTimezoneName", NY)
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
    df = pd.DataFrame({"open": q["open"], "close": q["close"], "adjclose": adj}, index=idx, dtype=float)
    return df[~df.index.duplicated(keep="last")].dropna()


def _drop_forming_bar(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    now = pd.Timestamp.now(tz=NY)
    if len(idx) and idx[-1].date() == now.date() and (now.hour, now.minute) < MARKET_CLOSED_AFTER:
        return idx[:-1]
    return idx


def load_panels(tickers: list[str], start: str) -> Panels:
    bars = {}
    for t in tickers:
        bars[t] = _fetch(t, start)
        time.sleep(0.15)
    fx = _fetch(FX, start)["close"]

    cal = _drop_forming_bar(pd.DatetimeIndex(sorted(set().union(*(b.index for b in bars.values())))))
    fx = fx.reindex(fx.index.union(cal)).ffill().reindex(cal)

    def wide(field: str) -> pd.DataFrame:
        return pd.DataFrame({t: b[field] for t, b in bars.items()}).reindex(cal).ffill()

    adjclose, close, open_ = wide("adjclose"), wide("close"), wide("open")
    adj_open = open_ * (adjclose / close)
    return Panels(open_eur=adj_open.div(fx, axis=0), close_eur=adjclose.div(fx, axis=0), close_usd=adjclose)
