"""Run with: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness import backtest, metrics, validate  # noqa: E402
from harness.strategies import STRATEGIES, cheater, momentum, sma_cross  # noqa: E402


def random_walk(n: int = 2000, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.03, n)))
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)


class BacktestTest(unittest.TestCase):
    def test_position_is_previous_signal(self):
        df = random_walk(50)
        sig = pd.Series(np.arange(50) % 2, index=df.index, dtype=float)
        bt = backtest.run(df["close"], sig, backtest.Costs(0, 0))
        pd.testing.assert_series_equal(bt["pos"], sig.shift(1).fillna(0.0), check_names=False)

    def test_same_bar_knowledge_earns_nothing_extra(self):
        # Knowing bar t's return at its close is fine; it must not earn bar t's return.
        df = random_walk(3000, seed=1)
        ret = df["close"].pct_change()
        honest = backtest.run(df["close"], (ret > 0).astype(float), backtest.Costs(0, 0))
        self.assertLess(abs(metrics.sharpe(honest["net"])), 1.0)

    def test_costs_charged_per_unit_turnover(self):
        df = random_walk(11)
        flip = pd.Series([1.0, -1.0] * 5 + [1.0], index=df.index)
        bt = backtest.run(df["close"], flip, backtest.Costs(fee_bps=10, slippage_bps=0))
        # first bar enters (turnover 1), then every bar flips -1 <-> 1 (turnover 2)
        self.assertAlmostEqual(bt["turnover"].sum(), 1 + 2 * 9)
        self.assertAlmostEqual(bt["cost"].sum(), (1 + 2 * 9) * 0.001)


class LookaheadTest(unittest.TestCase):
    def test_catches_cheater(self):
        self.assertTrue(validate.find_lookahead(cheater, random_walk(), {}))

    def test_passes_honest_strategies(self):
        df = random_walk()
        self.assertEqual(validate.find_lookahead(sma_cross, df, {"fast": 20, "slow": 100}), [])
        self.assertEqual(validate.find_lookahead(momentum, df, {"lookback": 30}), [])

    def test_cheater_fails_evaluation_despite_huge_sharpe(self):
        fn, grid = STRATEGIES["cheater"]
        r = validate.evaluate(fn, random_walk(), grid)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertFalse(r["gates"]["no_lookahead"])
        self.assertGreater(r["in_sample_best"]["sharpe"], validate.SUSPICIOUS_SHARPE)
        self.assertTrue(r["warnings"])


class DeflatedSharpeTest(unittest.TestCase):
    def test_single_trial_equals_psr_against_zero(self):
        r = random_walk(1000, drift=0.002)["close"].pct_change().dropna()
        self.assertAlmostEqual(metrics.deflated_sharpe(r, [0.1]), metrics.probabilistic_sharpe(r, 0.0))

    def test_more_trials_raise_the_bar(self):
        r = random_walk(1000, drift=0.002)["close"].pct_change().dropna()
        trials = list(np.random.default_rng(2).normal(0, 0.03, 200))
        self.assertLess(metrics.deflated_sharpe(r, trials), metrics.deflated_sharpe(r, trials[:3]))

    def test_noise_does_not_pass(self):
        # A driftless random walk has no edge; the gates should reject every example strategy on it.
        for name in ("sma_cross", "momentum"):
            fn, grid = STRATEGIES[name]
            r = validate.evaluate(fn, random_walk(2000, seed=3), grid)
            self.assertEqual(r["verdict"], "FAIL", name)


if __name__ == "__main__":
    unittest.main()
