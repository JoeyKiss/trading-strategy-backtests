"""Backtest every day-trading strategy on 5-minute bars (~60 days) and hourly bars (~3 years).

    python scripts/daytrade_backtest.py              # download fresh bars (merged into the cache), run all
    python scripts/daytrade_backtest.py --no-refresh # cached bars only

A strategy PASSES only if: no look-ahead, deflated Sharpe >= 0.95 (adjusted for how many variants were
tried), profitable after costs in both halves of the sample, and profitable on >= 60% of symbols.
"""
import argparse
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from daytrade.data import load, to_days  # noqa: E402
from daytrade.paper import SYMBOL_COSTS_BPS  # noqa: E402
from daytrade.sim import find_lookahead, run_days  # noqa: E402
from daytrade.strategies import CATALOG, LONG_ONLY  # noqa: E402
from harness import metrics  # noqa: E402
from papertrade import report  # noqa: E402
from papertrade.engine import Account  # noqa: E402

SYMBOLS = SYMBOL_COSTS_BPS  # symbol -> cost per side in bps
DATASETS = {"5m": "5-minute bars", "60m": "hourly bars"}
HOURLY_VARIANTS = {"orb_breakout": [{"range_min": 60}]}  # 15/30-minute ranges don't exist on hourly bars
OUT = ROOT / "data" / "daytrade"
YEAR = 252


def configs(interval: str):
    for name, (fn, variants, hourly_ok) in CATALOG.items():
        if interval == "60m":
            if not hourly_ok:
                continue
            variants = HOURLY_VARIANTS.get(name, variants)
        for p in variants:
            for long_only in ([True] if name in LONG_ONLY else [False, True]):
                label = name + (f"({','.join(f'{k}={v}' for k, v in p.items())})" if p else "")
                if long_only and name not in LONG_ONLY:
                    label += " long-only"
                yield label, fn, p, long_only


def evaluate(interval: str, refresh: bool) -> tuple[pd.DataFrame, dict]:
    print(f"\n[{DATASETS[interval]}] loading {len(SYMBOLS)} symbols...", flush=True)
    data = {s: to_days(load(s, interval, refresh)) for s in SYMBOLS}
    probe_days, probe_prev, _ = data["SPY"]
    rows, curves = [], {}
    for label, fn, p, long_only in configs(interval):
        leak = any(find_lookahead(fn, p, d, pc) for d, pc in zip(probe_days[-5:], probe_prev[-5:]))
        net, gross, trades = {}, {}, {}
        for s, (days, prev, dates) in data.items():
            r = run_days(fn, p, days, prev, SYMBOLS[s], long_only)
            gross[s], net[s], trades[s] = (pd.Series(r[:, j], index=dates) for j in range(3))  # run_days order
        net, gross, trades = pd.DataFrame(net), pd.DataFrame(gross), pd.DataFrame(trades)
        avg = net.mean(axis=1)  # average of the symbols' daily returns
        n, half = len(avg), len(avg) // 2
        n_trades = trades.sum().sum()
        rows.append({
            "strategy": label, "days": n, "leak": bool(leak),
            "net_yr": avg.mean() * YEAR, "gross_yr": gross.mean(axis=1).mean() * YEAR,
            "sharpe": metrics.sharpe(avg, YEAR), "t": avg.mean() / avg.std() * np.sqrt(n) if avg.std() else 0.0,
            "max_dd": metrics.max_drawdown(avg),
            "trades_day": n_trades / (n * len(SYMBOLS)),
            "gross_bps": gross.sum().sum() / n_trades * 1e4 if n_trades else 0.0,
            "net_bps": net.sum().sum() / n_trades * 1e4 if n_trades else 0.0,
            "sym_pos": (net.sum() > 0).mean(),
            "h1": metrics.sharpe(avg.iloc[:half], YEAR), "h2": metrics.sharpe(avg.iloc[half:], YEAR),
            "_sr": metrics.sharpe(avg, 1),
        })
        curves[label] = avg
    df = pd.DataFrame(rows)
    df["dsr"] = [metrics.deflated_sharpe(curves[s], list(df["_sr"])) for s in df["strategy"]]
    df["pass"] = (~df["leak"]) & (df["dsr"] >= 0.95) & (df["h1"] > 0) & (df["h2"] > 0) & (df["sym_pos"] >= 0.6)
    return df.drop(columns="_sr").sort_values("sharpe", ascending=False), curves


def print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}  ({df['days'].iloc[0]} sessions, {len(df)} variants tried)")
    print(f"{'strategy':40} {'net/yr':>7} {'gross/yr':>8} {'Sharpe':>6} {'t':>5} {'maxDD':>6} {'trd/d':>5} "
          f"{'bps gross>net':>14} {'sym+':>4} {'H1':>5} {'H2':>5} {'DSR':>5}")
    for r in df.itertuples():
        flag = " LEAK" if r.leak else (" PASS" if r.pass_ else "")
        print(f"{r.strategy:40} {r.net_yr:>7.1%} {r.gross_yr:>8.1%} {r.sharpe:>6.2f} {r.t:>5.1f} {r.max_dd:>6.1%} "
              f"{r.trades_day:>5.2f} {r.gross_bps:>6.1f} >{r.net_bps:>6.1f} {r.sym_pos:>4.0%} {r.h1:>5.1f} "
              f"{r.h2:>5.1f} {r.dsr:>5.2f}{flag}")


def html_section(df: pd.DataFrame, curves: dict, title: str) -> str:
    top = list(df["strategy"].head(5))
    if "open_to_close" not in top:
        top.append("open_to_close")
    accts = []
    for name in top:
        eq = 20_000 * (1 + curves[name]).cumprod()
        accts.append(Account(name=name, capital=20_000, cash=0,
                             equity=[[d.date().isoformat(), float(v)] for d, v in eq.items()]))
    cell = lambda v, fmt: f'<span class="{"pos" if v > 0 else "neg" if v < 0 else ""}">{format(v, fmt)}</span>'
    rows = "".join(
        f"<tr><td>{escape(r.strategy)}{' <b>PASS</b>' if r.pass_ else ''}{' <b>LEAK</b>' if r.leak else ''}</td>"
        f"<td>{cell(r.net_yr, '.1%')}</td><td>{cell(r.gross_yr, '.1%')}</td><td>{r.sharpe:.2f}</td>"
        f"<td>{r.max_dd:.1%}</td><td>{r.trades_day:.2f}</td><td>{r.gross_bps:.1f} → {cell(r.net_bps, '.1f')}</td>"
        f"<td>{r.sym_pos:.0%}</td><td>{r.h1:.1f} / {r.h2:.1f}</td><td>{r.dsr:.2f}</td></tr>"
        for r in df.rename(columns={"pass": "pass_"}).itertuples())
    return (f"<h2>{escape(title)}</h2><div class='card'>{report._chart(accts)}</div><div class='card' style='margin-top:12px'>"
            "<table><tr><th>Strategy</th><th>Net / yr</th><th>Gross / yr</th><th>Sharpe</th><th>Max DD</th>"
            "<th>Trades / day</th><th>bps / trade</th><th>Symbols +</th><th>Sharpe H1 / H2</th><th>DSR</th></tr>"
            f"{rows}</table></div>")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-refresh", action="store_true")
    a = p.parse_args()
    sections = []
    for interval, name in DATASETS.items():
        df, curves = evaluate(interval, not a.no_refresh)
        print_table(df.rename(columns={"pass": "pass_"}), name)
        days = curves[df["strategy"].iloc[0]].index
        title = f"{name}: {days[0].date()} to {days[-1].date()}, {len(days)} sessions"
        sections.append(html_section(df, curves, title))
        print(f"\n  -> {int(df['pass'].sum())} of {len(df)} variants pass every gate")
    OUT.mkdir(parents=True, exist_ok=True)
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Day-trading backtest</title><style>{report.CSS}"
            "</style></head><body><main><h1>Day-trading backtest</h1><p class='sub'>Average of 12 symbols, "
            "EUR 20k each, flat every night. Costs: 2 bps/side ETFs, 4 bps/side stocks. Equity curves: top 5 "
            f"by Sharpe plus the buy-open/sell-close baseline.</p>{''.join(sections)}</main></body></html>")
    (OUT / "report.html").write_text(page, encoding="utf-8")
    print(f"\nReport: {OUT / 'report.html'}")


if __name__ == "__main__":
    main()
