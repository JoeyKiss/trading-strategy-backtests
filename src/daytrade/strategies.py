"""Day-trading strategies. Each: (day, previous close, **params) -> trades. All positions are flat by
the close. Indicators are computed within the session only (they restart every morning)."""
import numpy as np

from .sim import Day, Trade, positions_to_trades


def _bar_index_at(d: Day, minute: int) -> int | None:
    hits = np.nonzero(d.mins >= minute)[0]
    return int(hits[0]) if len(hits) else None


def open_to_close(d: Day, prev_close, **_) -> list[Trade]:
    """Baseline: buy the open, sell the close. Any long strategy should beat simply doing this."""
    return [Trade(entry=0, side=1)]


def orb_first_bar(d: Day, prev_close, target_r: float = 10.0) -> list[Trade]:
    """Opening Range Breakout, Zarattini & Aziz (2023): trade in the direction of the first bar, stop at
    that bar's opposite extreme, target 10x the risk, otherwise exit at the close."""
    if len(d.open) < 2 or d.close[0] == d.open[0]:
        return []
    side = 1 if d.close[0] > d.open[0] else -1
    stop = d.low[0] if side > 0 else d.high[0]
    risk = abs(d.open[1] - stop)
    if risk == 0 or (side > 0 and d.open[1] <= stop) or (side < 0 and d.open[1] >= stop):
        return []
    return [Trade(entry=1, side=side, stop=stop, target=d.open[1] + side * target_r * risk)]


def orb_breakout(d: Day, prev_close, range_min: int = 30) -> list[Trade]:
    """Classic opening-range breakout: mark the high/low of the first `range_min` minutes, then trade
    the first close outside that range, stop at the other side, exit at the close. One trade a day."""
    rng = d.mins < range_min
    if rng.all() or not rng.any():
        return []
    hi, lo = d.high[rng].max(), d.low[rng].min()
    for i in np.nonzero(~rng)[0][:-1]:
        if d.close[i] > hi:
            return [Trade(entry=int(i) + 1, side=1, stop=lo)]
        if d.close[i] < lo:
            return [Trade(entry=int(i) + 1, side=-1, stop=hi)]
    return []


def vwap_trend(d: Day, prev_close) -> list[Trade]:
    """Zarattini & Aziz (2023) VWAP strategy: long while price closes above the session VWAP, short
    below, flipping at every cross."""
    typical = (d.high + d.low + d.close) / 3
    cum_v = np.cumsum(d.volume)
    vwap = np.divide(np.cumsum(typical * d.volume), cum_v, out=np.full(len(cum_v), np.nan), where=cum_v > 0)
    pos = np.where(np.isnan(vwap), 0, np.sign(d.close - vwap))
    return positions_to_trades(pos)


def intraday_momentum(d: Day, prev_close, first_min: int = 30, last_min: int = 360) -> list[Trade]:
    """Gao, Han, Li & Zhou (2018), 'Market intraday momentum': the return from yesterday's close to
    ~10:00 predicts the last half hour. Enter at 15:30 in that direction, exit at the close.
    (On hourly bars the first bar closes at 10:30, so the signal uses that.)"""
    if prev_close is None:
        return []
    early = np.nonzero(d.mins < first_min)[0]
    entry = _bar_index_at(d, last_min)
    if not len(early) or entry is None or entry <= early[-1]:
        return []
    r = d.close[early[-1]] / prev_close - 1
    return [Trade(entry=entry, side=1 if r > 0 else -1)] if r != 0 else []


def gap_fade(d: Day, prev_close, threshold: float = 0.005) -> list[Trade]:
    """Bet that an opening gap closes: trade toward yesterday's close at the open, target = yesterday's
    close, stop if the gap doubles."""
    if prev_close is None:
        return []
    g = d.open[0] / prev_close - 1
    if abs(g) < threshold:
        return []
    return [Trade(entry=0, side=-1 if g > 0 else 1, target=prev_close, stop=d.open[0] * (1 + g))]


def gap_go(d: Day, prev_close, threshold: float = 0.005) -> list[Trade]:
    """The opposite bet: a big gap keeps going. Trade with the gap, stop at yesterday's close."""
    if prev_close is None:
        return []
    g = d.open[0] / prev_close - 1
    if abs(g) < threshold:
        return []
    return [Trade(entry=0, side=1 if g > 0 else -1, stop=prev_close)]


def bollinger_revert(d: Day, prev_close, n: int = 20, k: float = 2.0) -> list[Trade]:
    """Mean reversion: buy a close below the lower Bollinger band, sell at the middle band; mirror for
    shorts above the upper band."""
    c = d.close
    pos, cur = np.zeros(len(c)), 0
    for i in range(n - 1, len(c)):
        w = c[i - n + 1:i + 1]
        mid, sd = w.mean(), w.std()
        if cur == 0 and sd > 0:
            cur = 1 if c[i] < mid - k * sd else -1 if c[i] > mid + k * sd else 0
        elif (cur == 1 and c[i] >= mid) or (cur == -1 and c[i] <= mid):
            cur = 0
        pos[i] = cur
    return positions_to_trades(pos)


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a, out = 2 / (n + 1), np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def ema_cross(d: Day, prev_close, fast: int = 9, slow: int = 21) -> list[Trade]:
    """The classic retail setup: long while the 9-bar EMA is above the 21-bar EMA, short below."""
    f, s = _ema(d.close, fast), _ema(d.close, slow)
    pos = np.sign(f - s)
    pos[:slow] = 0
    return positions_to_trades(pos)


def rsi_revert(d: Day, prev_close, n: int = 14, low: float = 30, high: float = 70) -> list[Trade]:
    """RSI mean reversion: buy when RSI(14) < 30 and exit when it recovers above 50; short above 70."""
    delta = np.diff(d.close, prepend=d.close[0])
    gain, loss = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    pos, cur, ag, al = np.zeros(len(d.close)), 0, 0.0, 0.0
    for i in range(1, len(d.close)):
        ag, al = ag + (gain[i] - ag) / n, al + (loss[i] - al) / n  # Wilder smoothing
        if i < n or ag + al == 0:
            continue
        rsi = 100 * ag / (ag + al)
        if cur == 0:
            cur = 1 if rsi < low else -1 if rsi > high else 0
        elif (cur == 1 and rsi > 50) or (cur == -1 and rsi < 50):
            cur = 0
        pos[i] = cur
    return positions_to_trades(pos)


def donchian(d: Day, prev_close, n: int = 12) -> list[Trade]:
    """Turtle-style channel breakout: long on a close above the prior n-bar high, out on a close below
    the prior n/2-bar low; mirror for shorts."""
    pos, cur, m = np.zeros(len(d.close)), 0, max(n // 2, 1)
    for i in range(n, len(d.close)):
        c = d.close[i]
        if cur == 0:
            cur = 1 if c > d.high[i - n:i].max() else -1 if c < d.low[i - n:i].min() else 0
        elif (cur == 1 and c < d.low[i - m:i].min()) or (cur == -1 and c > d.high[i - m:i].max()):
            cur = 0
        pos[i] = cur
    return positions_to_trades(pos)


# name -> (function, parameter variants, works on hourly bars?). Every variant x direction is a trial.
CATALOG = {
    "open_to_close": (open_to_close, [{}], True),
    "orb_first_bar": (orb_first_bar, [{}], True),
    "orb_breakout": (orb_breakout, [{"range_min": 15}, {"range_min": 30}, {"range_min": 60}], True),
    "vwap_trend": (vwap_trend, [{}], True),
    "intraday_momentum": (intraday_momentum, [{}], True),
    "gap_fade": (gap_fade, [{"threshold": 0.005}, {"threshold": 0.01}], True),
    "gap_go": (gap_go, [{"threshold": 0.005}, {"threshold": 0.01}], True),
    "bollinger_revert": (bollinger_revert, [{}], False),
    "ema_cross": (ema_cross, [{}], False),
    "rsi_revert": (rsi_revert, [{}], False),
    "donchian": (donchian, [{}], False),
}
LONG_ONLY = {"open_to_close"}  # baseline, has no short version
