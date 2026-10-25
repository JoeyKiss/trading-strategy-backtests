"""Run with: python -m unittest discover tests  (no network: the API is stubbed)"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from daytrade import alpaca  # noqa: E402
from daytrade import data as yahoo  # noqa: E402
from daytrade.sim import Day, Trade, run_days  # noqa: E402


def bar(t, c=100.0):
    return {"t": t, "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 1000, "n": 5, "vw": c}


class SessionTest(unittest.TestCase):
    def test_utc_to_new_york_across_dst(self):
        # 09:30 New York is 14:30 UTC in winter (EST) and 13:30 UTC in summer (EDT).
        pages = [{"bars": {"SPY": [bar("2016-01-04T14:25:00Z"), bar("2016-01-04T14:30:00Z"),
                                   bar("2016-07-05T13:30:00Z"), bar("2016-07-05T19:55:00Z"),
                                   bar("2016-07-05T20:00:00Z")]}, "next_page_token": None}]
        df = alpaca.regular_session(alpaca.fetch_bars("SPY", "2016-01-01", pd.Timestamp("2016-08-01", tz="UTC"),
                                                      get=lambda p, k: pages.pop(0)))
        self.assertEqual([str(t) for t in df.index],
                         ["2016-01-04 09:30:00", "2016-07-05 09:30:00", "2016-07-05 15:55:00"])  # 09:25 and 16:00 dropped

    def test_early_close_days_end_at_13(self):
        # NYSE 13:00 closes; SIP bars after that are after-hours trades and must not count as the session
        self.assertEqual(sorted(str(d.date()) for d in alpaca.early_closes([2019, 2020, 2026])),
                         ["2019-07-03", "2019-11-29", "2019-12-24", "2020-11-27", "2020-12-24", "2026-11-27",
                          "2026-12-24"])
        pages = [{"bars": {"SPY": [bar("2019-11-29T17:55:00Z"), bar("2019-11-29T18:00:00Z"),
                                   bar("2019-12-02T20:55:00Z")]}, "next_page_token": None}]
        df = alpaca.regular_session(alpaca.fetch_bars("SPY", "2019-11-01", pd.Timestamp("2019-12-03", tz="UTC"),
                                                      get=lambda p, k: pages.pop(0)))
        self.assertEqual([str(t) for t in df.index], ["2019-11-29 12:55:00", "2019-12-02 15:55:00"])

    def test_no_previous_close_after_broken_session(self):
        idx = pd.DatetimeIndex(["2018-05-01 09:30", "2018-05-01 09:35", "2018-05-01 09:40", "2018-05-02 09:30",
                                "2018-05-03 09:30", "2018-05-03 09:35", "2018-05-03 09:40"])
        df = pd.DataFrame({c: np.arange(7.0) + 1 for c in alpaca.COLS}, index=idx)
        days, prev, dates = yahoo.to_days(df)
        self.assertEqual([str(d.date()) for d in dates], ["2018-05-01", "2018-05-03"])
        self.assertEqual(prev, [None, None])  # not the lone 09:30 bar of the outage day

    def test_follows_page_tokens(self):
        calls = []

        def get(params, keys):
            calls.append(params.get("page_token"))
            if len(calls) == 1:
                return {"bars": {"SPY": [bar("2016-01-04T14:30:00Z")]}, "next_page_token": "abc"}
            return {"bars": {"SPY": [bar("2016-01-04T14:35:00Z")]}, "next_page_token": None}
        df = alpaca.fetch_bars("SPY", "2016-01-01", pd.Timestamp("2016-02-01", tz="UTC"), get=get)
        self.assertEqual(calls, [None, "abc"])
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), alpaca.COLS)

    def test_empty_response(self):
        df = alpaca.fetch_bars("SPY", "2016-01-01", pd.Timestamp("2016-01-02", tz="UTC"),
                               get=lambda p, k: {"bars": {}, "next_page_token": None})
        self.assertTrue(df.empty)


class EnvTest(unittest.TestCase):
    def test_formats_editors_save(self):
        text = "# comment\nexport ALPACA_API_KEY=abc\nALPACA_SECRET_KEY = \"x=y\"\n\n"
        for raw in (text.encode("utf-8"), b"\xef\xbb\xbf" + text.encode("utf-8"), text.encode("utf-16")):
            self.assertEqual(alpaca.parse_env(raw), {"ALPACA_API_KEY": "abc", "ALPACA_SECRET_KEY": "x=y"})


class ParityTest(unittest.TestCase):
    def test_parity_metric(self):
        idx = pd.date_range("2026-10-01 09:30", periods=78, freq="5min")
        y = pd.DataFrame({"close": np.linspace(100, 101, 78)}, index=idx)
        a = y * 0.997  # same bars, scaled by a later dividend adjustment
        p = alpaca.parity(y, a)
        self.assertEqual(int(p["shared_bars"].iloc[0]), 78)
        self.assertAlmostEqual(p["median_ratio"].iloc[0], 0.997)
        self.assertLess(p["ratio_spread"].iloc[0], 1e-9)

    @unittest.skipUnless((alpaca.CACHE / "SPY_5Min.parquet").exists() and (yahoo.CACHE / "SPY_5m.csv").exists(),
                         "needs both the Alpaca and the Yahoo SPY caches")
    def test_real_caches_agree(self):
        p = alpaca.parity(yahoo.load("SPY", "5m", refresh=False), alpaca.load("SPY", max_age_days=float("inf")))
        self.assertGreater(len(p), 20)
        self.assertGreater((p["yahoo_bars"] == p["alpaca_bars"]).mean(), 0.9)  # same sessions, same bars
        self.assertLess(p["ratio_spread"].median(), 0.001)                      # closes agree within ~0.1%
        self.assertLess(abs(p["median_ratio"].median() - 1), 0.02)               # only dividend scaling apart


class PerDayCostTest(unittest.TestCase):
    def test_costs_can_vary_by_session(self):
        p = np.array([100.0, 100, 100])
        d = Day(p, p, p, p, np.ones(3), np.arange(3) * 5)
        out = run_days(lambda d, pc: [Trade(0, 1)], {}, [d, d], [None, None], cost_bps=[1.0, 10.0])
        np.testing.assert_allclose(out[:, 1], [-2 / 1e4, -20 / 1e4])


if __name__ == "__main__":
    unittest.main()
