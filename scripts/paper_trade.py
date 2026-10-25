"""Paper trading with EUR 20k per strategy. Same engine for history replays and live forward trading.

    python scripts/paper_trade.py backtest --start 2018-01-01   # replay history, print the gates
    python scripts/paper_trade.py init                          # open live paper accounts as of the last close
    python scripts/paper_trade.py init-daytrade --start 2026-10-07  # open the day-trading account
    python scripts/paper_trade.py run                           # process new trading days (run once a day)
    python scripts/paper_trade.py report                        # rebuild the dashboard

Orders are decided at each close and filled at the next open, with EUR 1 per order plus 10 bps.
The day-trading account trades on 5-minute bars and is flat every night (see src/daytrade/paper.py).
Every `run` also refreshes the 5-minute cache, so intraday history keeps growing past Yahoo's 60 days.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from daytrade import paper as daytrade  # noqa: E402
from harness import metrics  # noqa: E402
from papertrade import engine, market, report  # noqa: E402
from papertrade.strategies import BENCHMARK_STRATEGY, BENCHMARKS, STRATEGIES, TICKERS  # noqa: E402

PAPER_DIR = ROOT / "data" / "paper"
STATE = PAPER_DIR / "state.json"
DAY_STATE = PAPER_DIR / "daytrade_state.json"
DASHBOARD = PAPER_DIR / "dashboard.html"
WARMUP = pd.DateOffset(years=2)  # history strategies need before their first decision
TRADING_DAYS = 252


def _panels(since: str) -> market.Panels:
    print(f"Fetching {len(TICKERS)} tickers + EUR/USD from Yahoo...", flush=True)
    return market.load_panels(TICKERS, (pd.Timestamp(since) - WARMUP).date().isoformat())


def _print_summary(accounts: list[engine.Account]) -> None:
    bench = next(a for a in accounts if a.name == BENCHMARK_STRATEGY)
    b = bench.value / bench.capital - 1
    print(f"\n{'strategy':16} {'value':>10} {'return':>8} {'vs bench':>9} {'trades':>7} {'costs':>8}")
    for a in accounts:
        r = a.value / a.capital - 1
        vs = "" if a is bench else f"{r - b:+.1%}"
        print(f"{a.name:16} {a.value:>10,.0f} {r:>+8.1%} {vs:>9} {len(a.trades):>7} {a.fees:>8,.0f}")


def _with_daytrade(accounts: list[engine.Account]) -> list[engine.Account]:
    return accounts + engine.load(DAY_STATE) if DAY_STATE.exists() else accounts


def _write_dashboard(accounts, path: Path, title: str) -> None:
    span = f"{accounts[0].equity[0][0]} to {accounts[0].equity[-1][0]}"
    sub = f"{span} · EUR {accounts[0].capital:,.0f} per strategy · fills at next open, EUR 1 + 0.10% per order"
    if any(a.name == daytrade.NAME for a in accounts):
        sub += f" · {daytrade.NAME}: 5-minute bars, flat every night, EUR 1 + 2-4 bps per order"
    path.write_text(report.render(accounts, title, sub, BENCHMARK_STRATEGY), encoding="utf-8")
    print(f"\nDashboard: {path}")


def cmd_backtest(a) -> None:
    panels = _panels(a.start)
    accounts = []
    for name, fn in STRATEGIES.items():
        acct = engine.open_account(name, a.capital, fn, panels, a.start)
        engine.advance(acct, fn, panels)
        accounts.append(acct)
    _print_summary(accounts)

    rets = {acct.name: engine.returns(acct) for acct in accounts}
    trial_sharpes = [metrics.sharpe(rets[n], 1) for n in rets if n not in BENCHMARKS]
    print(f"\n{'strategy':16} {'Sharpe':>7} {'CAGR':>7} {'max DD':>7} {'deflated Sharpe':>16}")
    for n, r in rets.items():
        dsr = "" if n in BENCHMARKS else f"{metrics.deflated_sharpe(r, trial_sharpes):.3f}"
        print(f"{n:16} {metrics.sharpe(r, TRADING_DAYS):>7.2f} {metrics.cagr(r, TRADING_DAYS):>7.1%} "
              f"{metrics.max_drawdown(r):>7.1%} {dsr:>16}")
    print("\nCaveat: the stock list is today's big winners, picked with hindsight (survivorship bias), so every"
          "\nstock strategy looks better in this replay than it would have live. Judge stock strategies against"
          "\nuniverse_equal, not world_buy_hold. Forward paper trading has no such bias.")
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    _write_dashboard(accounts, PAPER_DIR / "backtest.html", "Paper trading: historical replay")


def cmd_init(a) -> None:
    if STATE.exists() and not a.force:
        sys.exit(f"{STATE} already exists. Use --force to wipe it and start over.")
    panels = _panels(a.start or pd.Timestamp.today().date().isoformat())
    start = a.start or panels.dates[-1].date().isoformat()
    accounts = [engine.open_account(n, a.capital, fn, panels, start) for n, fn in STRATEGIES.items()]
    start = accounts[0].last_date  # snapped to the last trading day on or before --start
    engine.save(accounts, STATE)
    print(f"Opened {len(accounts)} accounts as of the {start} close. First orders fill at the next open:")
    for acct in accounts:
        print(f"  {acct.name:16} {', '.join(sorted(acct.pending)) or 'cash'}")
    _write_dashboard(accounts, DASHBOARD, "Paper trading: live")


def cmd_run(a) -> None:
    if not STATE.exists():
        sys.exit("No paper accounts yet. Run: python scripts/paper_trade.py init")
    accounts = engine.load(STATE)
    panels = _panels(min(acct.last_date for acct in accounts))
    days = 0
    for acct in accounts:
        days = engine.advance(acct, STRATEGIES[acct.name], panels)
    engine.save(accounts, STATE)
    print(f"Processed {days} new trading day(s), now as of {accounts[0].last_date}.")

    print(f"Refreshing 5-minute bars for {len(daytrade.SYMBOL_COSTS_BPS)} symbols...", flush=True)
    sessions = daytrade.load_sessions(refresh=True)  # also grows the intraday cache
    if DAY_STATE.exists():
        (day_acct,) = engine.load(DAY_STATE)
        n = daytrade.advance(day_acct, sessions)
        engine.save([day_acct], DAY_STATE)
        print(f"Day-trading account: {n} new session(s), now as of {day_acct.last_date}.")
    accounts = _with_daytrade(accounts)
    _print_summary(accounts)
    _write_dashboard(accounts, DASHBOARD, "Paper trading: live")


def cmd_init_daytrade(a) -> None:
    if DAY_STATE.exists() and not a.force:
        sys.exit(f"{DAY_STATE} already exists. Use --force to wipe it and start over.")
    acct = daytrade.open_account(a.capital, a.start)
    n = daytrade.advance(acct, daytrade.load_sessions(refresh=True))
    engine.save([acct], DAY_STATE)
    print(f"Opened {daytrade.NAME} with EUR {a.capital:,.0f}; booked {n} session(s) since {a.start}.")
    for t in acct.trades:
        print(f"  {t['date']} {t['side']} {t['ticker']:5} EUR {t['eur']:,.0f} -> P&L {t['pnl']:+.2f} ({t['exit']})")
    if STATE.exists():
        cmd_report(a)


def cmd_report(a) -> None:
    _write_dashboard(_with_daytrade(engine.load(STATE)), DASHBOARD, "Paper trading: live")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--start", default="2018-01-01")
    b.add_argument("--capital", type=float, default=20_000)
    i = sub.add_parser("init")
    i.add_argument("--capital", type=float, default=20_000)
    i.add_argument("--force", action="store_true")
    i.add_argument("--start", help="open at this day's close (default: the latest close); "
                                   "follow with `run` to process the days since")
    dt = sub.add_parser("init-daytrade")
    dt.add_argument("--start", required=True, help="first session to trade, e.g. 2026-10-07")
    dt.add_argument("--capital", type=float, default=20_000)
    dt.add_argument("--force", action="store_true")
    sub.add_parser("run")
    sub.add_parser("report")
    a = p.parse_args()
    {"backtest": cmd_backtest, "init": cmd_init, "init-daytrade": cmd_init_daytrade, "run": cmd_run,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
