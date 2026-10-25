"""Run with: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daytrade.sim import Day, Trade, execute, find_lookahead, positions_to_trades, run_days  # noqa: E402
from daytrade.strategies import CATALOG  # noqa: E402


def day(n: int = 78, seed: int = 0, step: int = 5) -> Day:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.concatenate([[100.0], c[:-1]]) * np.exp(rng.normal(0, 0.0005, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.002, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.002, n))
    return Day(o, h, l, c, rng.uniform(1e5, 2e5, n), np.arange(n) * step)


def flat_day(prices) -> Day:
    p = np.asarray(prices, dtype=float)
    return Day(p, p, p, p, np.ones(len(p)), np.arange(len(p)) * 5)


class ExecuteTest(unittest.TestCase):
    def test_hold_to_close(self):
        d = flat_day([100, 101, 102, 103])
        (f,) = execute(d, [Trade(entry=1, side=1)])
        self.assertAlmostEqual(f["gross"], 103 / 101 - 1)

    def test_short_profits_when_price_falls(self):
        d = flat_day([100, 99, 98])
        (f,) = execute(d, [Trade(entry=0, side=-1)])
        self.assertAlmostEqual(f["gross"], -(98 / 100 - 1))

    def test_stop_fills_at_stop_or_worse_open(self):
        d = Day(np.array([100, 100, 95.0]), np.array([100, 100.5, 95]), np.array([100, 98, 94]),
                np.array([100, 99, 94.5]), np.ones(3), np.arange(3) * 5)
        (f,) = execute(d, [Trade(entry=1, side=1, stop=99)])
        self.assertEqual(f["exit"], 1)
        self.assertAlmostEqual(f["gross"], 99 / 100 - 1)  # stopped inside bar 1 at 99
        (g,) = execute(d, [Trade(entry=1, side=1, stop=96, exit=None)])
        self.assertAlmostEqual(g["gross"], 95 / 100 - 1)  # gapped through 96: filled at the 95 open

    def test_stop_beats_target_in_same_bar(self):
        d = Day(np.array([100, 100.0]), np.array([100, 105]), np.array([100, 95]), np.array([100, 100]),
                np.ones(2), np.arange(2) * 5)
        (f,) = execute(d, [Trade(entry=1, side=1, stop=97, target=103)])
        self.assertAlmostEqual(f["gross"], 97 / 100 - 1)

    def test_flips_enter_on_the_exit_bar(self):
        trades = positions_to_trades(np.array([1, 1, -1, -1, 0, 0]))
        self.assertEqual(trades, [Trade(1, 1, exit=3), Trade(3, -1, exit=5)])
        self.assertEqual(len(execute(flat_day(range(100, 106)), trades)), 2)

    def test_costs_per_round_trip(self):
        d = flat_day([100, 100, 100])
        out = run_days(lambda d, pc: [Trade(0, 1)], {}, [d], [None], cost_bps=3)
        self.assertAlmostEqual(out[0, 1], -6 / 1e4)


class LookaheadTest(unittest.TestCase):
    def test_every_strategy_is_clean(self):
        for name, (fn, variants, _) in CATALOG.items():
            for p in variants:
                for seed in range(3):
                    d = day(seed=seed)
                    self.assertEqual(find_lookahead(fn, p, d, 99.0), [], f"{name} {p}")

    def test_detector_catches_peeking(self):
        def cheat(d, pc):  # decides bar e's trade using bar e's close
            return [Trade(entry=i, side=1 if d.close[i] > d.open[i] else -1, exit=i + 1)
                    for i in range(len(d.open) - 1)]
        self.assertTrue(find_lookahead(cheat, {}, day(), 99.0))


class PaperAccountTest(unittest.TestCase):
    def sessions(self, symbols):
        """Session 1 ends at 100. Session 2 gaps down to 98 (-2%) and drifts back to 99."""
        import pandas as pd
        d1, d2 = flat_day([100, 100, 100]), flat_day([98, 98.5, 99])
        dates = [pd.Timestamp("2026-10-06"), pd.Timestamp("2026-10-07")]
        return {s: ([d1, d2], [None, 100.0], dates) for s in symbols}

    def test_gap_down_is_bought_and_booked_with_costs(self):
        from daytrade import paper
        acct = paper.open_account(20_000, "2026-10-06")
        self.assertEqual(paper.advance(acct, self.sessions(["SPY"])), 2)
        (t,) = acct.trades
        slot = 20_000 / paper.SLOTS
        cost = 2 * (1 + slot * (2 - 0.5) / 1e4)
        self.assertAlmostEqual(t["pnl"], round(slot * (99 / 98 - 1) - cost, 2), places=2)
        self.assertEqual(len(acct.equity), 3)  # start point + 2 sessions

    def test_at_most_slots_trades_a_day_and_no_double_booking(self):
        from daytrade import paper
        acct = paper.open_account(20_000, "2026-10-06")
        s = self.sessions(["AAPL", "AMD", "AMZN", "GOOGL", "META"])
        paper.advance(acct, s)
        self.assertEqual(len(acct.trades), paper.SLOTS)
        value = acct.cash
        self.assertEqual(paper.advance(acct, s), 0)  # rerunning the same data books nothing new
        self.assertEqual(acct.cash, value)


if __name__ == "__main__":
    unittest.main()
