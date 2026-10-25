"""Run with: python -m unittest discover tests"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from papertrade import engine  # noqa: E402
from papertrade.market import Panels  # noqa: E402
from papertrade.strategies import rsi  # noqa: E402

FREE = engine.Costs(fee_eur=0, slippage_bps=0, min_trade_frac=0, cash_buffer=0)


def panels(n: int = 60, seed: int = 0) -> Panels:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, 2)), axis=0)), index=idx, columns=["A", "B"])
    open_ = close.shift(1).fillna(close) * np.exp(rng.normal(0, 0.005, (n, 2)))  # overnight gaps
    return Panels(open_eur=open_, close_eur=close, close_usd=close)


def all_in(ticker):
    return lambda closes, held: {ticker: 1.0}


class EngineTest(unittest.TestCase):
    def test_orders_fill_at_next_open_not_same_day(self):
        p = panels()
        acct = engine.open_account("x", 1000, all_in("A"), p, str(p.dates[0].date()))
        self.assertEqual(acct.positions, {})
        engine.advance(acct, all_in("A"), p, FREE)
        self.assertEqual(acct.trades[0]["date"], str(p.dates[1].date()))
        # Zero costs: bought at day-1 open, so final value = capital * last close / day-1 open.
        expected = 1000 * p.close_eur["A"].iloc[-1] / p.open_eur["A"].iloc[1]
        self.assertAlmostEqual(acct.value, expected, places=6)

    def test_strategy_only_sees_data_up_to_decision_day(self):
        p = panels()
        seen = []

        def spy(closes, held):
            seen.append(closes.index[-1])
            return {}
        acct = engine.open_account("x", 1000, spy, p, str(p.dates[0].date()))
        engine.advance(acct, spy, p, FREE)
        self.assertEqual(seen, list(p.dates))

    def test_costs_are_charged(self):
        p = panels()
        costs = engine.Costs(fee_eur=1, slippage_bps=10, min_trade_frac=0, cash_buffer=0.005)
        acct = engine.open_account("x", 1000, all_in("A"), p, str(p.dates[0].date()))
        engine.advance(acct, all_in("A"), p, costs)
        first = acct.trades[0]
        self.assertAlmostEqual(first["cost"], round(1 + first["eur"] * 0.001, 2), places=2)
        self.assertGreater(acct.cash, 0)  # the buffer keeps fees from overdrawing cash

    def test_switching_sells_then_buys(self):
        p = panels()
        flip = lambda closes, held: {"B": 1.0} if len(closes) % 2 else {"A": 1.0}
        acct = engine.open_account("x", 1000, flip, p, str(p.dates[0].date()))
        engine.advance(acct, flip, p, FREE)
        self.assertLessEqual(len(acct.positions), 1)
        self.assertAlmostEqual(acct.cash, 0, places=6)

    def test_resume_from_saved_state_matches_one_pass(self):
        p = panels()
        strat = lambda closes, held: {"A": 0.5, "B": 0.5} if closes["A"].iloc[-1] > closes["A"].mean() else {}
        one = engine.open_account("x", 1000, strat, p, str(p.dates[0].date()))
        engine.advance(one, strat, p)

        part = Panels(*(f.iloc[:30] for f in (p.open_eur, p.close_eur, p.close_usd)))
        two = engine.open_account("x", 1000, strat, part, str(p.dates[0].date()))
        engine.advance(two, strat, part)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.json"
            engine.save([two], path)
            (two,) = engine.load(path)
        engine.advance(two, strat, p)
        self.assertAlmostEqual(one.value, two.value, places=6)
        self.assertEqual(len(one.trades), len(two.trades))

    def test_rejects_bad_weights(self):
        p = panels()
        with self.assertRaises(ValueError):
            engine.open_account("x", 1000, lambda c, h: {"A": 0.8, "B": 0.8}, p, str(p.dates[0].date()))


class RsiTest(unittest.TestCase):
    def test_extremes(self):
        up = pd.DataFrame({"x": np.arange(1.0, 20.0)})
        self.assertAlmostEqual(rsi(up).iloc[-1, 0], 100.0)
        self.assertAlmostEqual(rsi(-up).iloc[-1, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
