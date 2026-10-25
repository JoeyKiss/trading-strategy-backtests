"""Run a strategy through every gate on real Coinbase data.

    python scripts/run_harness.py --strategy sma_cross
    python scripts/run_harness.py --strategy cheater          # should FAIL the look-ahead gate
    python scripts/run_harness.py --product ETH-USD --fee-bps 40
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness import backtest, data, validate  # noqa: E402
from harness.strategies import STRATEGIES  # noqa: E402


def fmt(s: dict) -> str:
    return (f"Sharpe {s['sharpe']:5.2f} | CAGR {s['cagr']:7.1%} | max DD {s['max_drawdown']:7.1%} | "
            f"costs/yr {s['cost_per_year']:6.2%} | in market {s['exposure']:4.0%}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="sma_cross")
    p.add_argument("--product", default="BTC-USD")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--train", type=int, default=730, help="walk-forward train window in bars")
    p.add_argument("--test", type=int, default=90, help="walk-forward test window in bars")
    p.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    a = p.parse_args()

    df = data.fetch_candles(a.product, a.start, refresh=a.refresh)
    fn, grid = STRATEGIES[a.strategy]
    costs = backtest.Costs(a.fee_bps, a.slippage_bps)
    r = validate.evaluate(fn, df, grid, costs, a.train, a.test)

    print(f"\n{a.product} daily, {df.index[0].date()} to {df.index[-1].date()} ({len(df)} bars), "
          f"costs {a.fee_bps + a.slippage_bps:g} bps per trade")
    print(f"Strategy: {r['strategy']} ({r['configs_tried']} parameter sets tried)\n")
    print(f"  In-sample best  : {fmt(r['in_sample_best'])}")
    if "out_of_sample" in r:
        print(f"  Walk-forward OOS: {fmt(r['out_of_sample'])}   (from {r['oos_start']})")
        print(f"  Buy & hold, OOS : {fmt(r['buy_and_hold'])}")
        print(f"  Deflated Sharpe : {r['deflated_sharpe']:.3f} (need >= {validate.MIN_DSR})")
    if "lookahead" in r:
        for params, bars in r["lookahead"].items():
            print(f"  Look-ahead found: params {params} changed at {len(bars)} checked bars, e.g. {bars[0].date()}")
    for w in r["warnings"]:
        print(f"  WARNING: {w}")
    print("\n  Gates: " + ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in r["gates"].items()))
    print(f"  VERDICT: {r['verdict']}\n")


if __name__ == "__main__":
    main()
