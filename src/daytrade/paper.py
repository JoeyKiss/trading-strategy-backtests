"""Forward paper trading for a day-trading strategy, booked into a papertrade Account so it appears on
the same dashboard. Each session: run the strategy on every symbol's 5-minute bars, give each trade one
of SLOTS equal slices of the account (first signals first, at most SLOTS trades a day), book the result.
Nothing is ever held overnight."""
import pandas as pd

from papertrade.engine import Account

from .data import load, to_days
from .sim import execute
from .strategies import gap_fade

# symbol -> cost per side in bps (spread + slippage, plus the EUR 1 fee as ~0.5 bp on EUR 20k)
SYMBOL_COSTS_BPS = {"SPY": 2, "QQQ": 2, "IWM": 2, "DIA": 2, "NVDA": 4, "TSLA": 4, "AAPL": 4, "MSFT": 4,
                    "AMZN": 4, "META": 4, "AMD": 4, "GOOGL": 4}
FEE_EUR = 1.0
FEE_BPS_INCLUDED = 0.5  # booked as a real EUR 1 per order here instead
SLOTS = 3

NAME = "daytrade_gapfade"
STRATEGY, PARAMS, LONG_ONLY = gap_fade, {"threshold": 0.01}, True  # the backtest's one candidate


def load_sessions(refresh: bool = True) -> dict:
    """5-minute sessions per symbol. Refreshing also grows the local cache past Yahoo's 60-day window."""
    return {s: to_days(load(s, "5m", refresh)) for s in SYMBOL_COSTS_BPS}


def open_account(capital: float, start: str) -> Account:
    """Trades every session from `start` on; the equity curve begins the business day before."""
    prev = (pd.Timestamp(start) - pd.offsets.BDay(1)).date().isoformat()
    acct = Account(name=NAME, capital=capital, cash=capital, last_date=prev)
    acct.equity.append([prev, round(capital, 2)])
    return acct


def advance(acct: Account, sessions: dict) -> int:
    """Book every completed session after acct.last_date. Returns the number of sessions processed."""
    by_date: dict[str, list] = {}
    for sym, (days, prevs, dates) in sessions.items():
        for d, pc, dt in zip(days, prevs, dates):
            if (iso := dt.date().isoformat()) > acct.last_date:
                by_date.setdefault(iso, []).append((sym, d, pc))
    for date in sorted(by_date):
        slot = acct.cash / SLOTS
        fills = []
        for sym, d, pc in by_date[date]:
            trades = [t for t in STRATEGY(d, pc, **PARAMS) if t.side > 0 or not LONG_ONLY]
            fills += [(f["entry"], sym, f) for f in execute(d, trades)]
        for _, sym, f in sorted(fills, key=lambda x: (x[0], x[1]))[:SLOTS]:
            cost = 2 * (FEE_EUR + slot * (SYMBOL_COSTS_BPS[sym] - FEE_BPS_INCLUDED) / 1e4)
            pnl = slot * f["gross"] - cost
            acct.cash += pnl
            acct.fees += cost
            acct.trades.append({"date": date, "ticker": sym, "side": "LONG" if f["side"] > 0 else "SHORT",
                                "eur": round(slot, 2), "cost": round(cost, 2), "pnl": round(pnl, 2),
                                "exit": f["how"]})
        acct.equity.append([date, round(acct.cash, 2)])
        acct.last_date = date
    return len(by_date)
