"""Day-trading backtest on Alpaca's full-market (SIP) 5-minute bars, 2016 to now: 12 symbols, 29 variants.

    python scripts/daytrade_longrun.py            # refreshes the Alpaca cache if it's older than 7 days
    python scripts/daytrade_longrun.py --cached   # never download

Out-of-sample design. The live paper account daytrade_gapfade (gap_fade, 1%, long-only) was picked on
Yahoo's 5-minute bars of 2026-07-15..2026-10-07 (the SELECTION window). Everything before that is data
the choice never saw, so 2016-01-04..2026-07-14 is the honest test (OOS). A walk-forward re-picks the
best variant every year using only the 3 years before it, to test the "pick the winner" process itself.

Costs per side: the table in daytrade/paper.py (2 bps ETFs, 4 bps stocks) plus half a cent per share,
because US stocks quote in whole cents (this matters for AMD at $2 in 2016, not for SPY at $700).
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from daytrade import alpaca, data as yahoo  # noqa: E402
from daytrade.data import to_days  # noqa: E402
from daytrade.paper import SYMBOL_COSTS_BPS  # noqa: E402
from daytrade.sim import find_lookahead, run_days  # noqa: E402
from daytrade.strategies import CATALOG, LONG_ONLY  # noqa: E402
from harness import metrics  # noqa: E402
from papertrade import report  # noqa: E402
from papertrade.engine import Account  # noqa: E402

SYMBOLS = list(SYMBOL_COSTS_BPS)
CANDIDATE = "gap_fade(threshold=0.01) long-only"
BASELINE = "open_to_close"
OOS_END = "2026-07-14"
SELECTION = ("2026-07-15", "2026-10-07")
HALF_TICK = 0.005  # USD per share
WF_TRAIN_YEARS = 3
YEAR = 252
OUT = ROOT / "data" / "daytrade"
LOG = OUT / "longrun.log"  # open it in the editor to watch progress live
DOWNLOAD_THREADS = 6  # requests are latency-bound (~4 s each); 6 in parallel stays far under 200/minute


class _Tee:
    """Everything printed also goes to LOG, timestamped, so a background run can be watched."""

    def __init__(self, stream, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream, self.file, self.fresh_line = stream, open(path, "w", encoding="utf-8"), True

    def write(self, s: str) -> int:
        self.stream.write(s)
        for part in s.splitlines(keepends=True):
            self.file.write((time.strftime("%H:%M:%S ") if self.fresh_line else "") + part)
            self.fresh_line = part.endswith("\n")
        self.file.flush()
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()


def variants():
    for name, (fn, params_list, _) in CATALOG.items():
        for p in params_list:
            for long_only in ([True] if name in LONG_ONLY else [False, True]):
                label = name + (f"({','.join(f'{k}={v}' for k, v in p.items())})" if p else "")
                if long_only and name not in LONG_ONLY:
                    label += " long-only"
                yield label, name, p, long_only


def load_universe(tick_floor: bool = True) -> dict:
    """symbol -> (days, prev closes, dates, cost per side in bps for each session). Cache only."""
    out = {}
    for s in SYMBOLS:
        days, prev, dates = to_days(alpaca.load(s, max_age_days=float("inf")))
        base = np.full(len(days), float(SYMBOL_COSTS_BPS[s]))
        if tick_floor:
            factor = alpaca.raw_factor(s).reindex(pd.DatetimeIndex(dates), method="ffill").fillna(1.0).to_numpy()
            raw_open = np.array([d.open[0] for d in days]) * factor
            base = base + 1e4 * HALF_TICK / raw_open
        out[s] = (days, prev, dates, base)
    return out


_U: dict = {}


def _init_worker(tick_floor: bool) -> None:
    _U.update(load_universe(tick_floor))


def _run_variant(v) -> tuple:
    label, name, p, long_only = v
    fn = CATALOG[name][0]
    days, prev, _, _ = _U["SPY"]
    leak = any(find_lookahead(fn, p, d, pc) for d, pc in zip(days[-5:], prev[-5:]))
    res = {}
    for s, (days, prev, dates, cost) in _U.items():
        r = run_days(fn, p, days, prev, cost, long_only)
        res[s] = pd.DataFrame(r, index=pd.DatetimeIndex(dates), columns=["gross", "net", "trades"])
    return label, leak, res


def summarize(label: str, res: dict, start=None, end=None, trial_srs=None) -> dict:
    sl = lambda df: df.loc[start:end]
    net = pd.DataFrame({s: sl(r)["net"] for s, r in res.items()})
    gross = pd.DataFrame({s: sl(r)["gross"] for s, r in res.items()})
    trades = pd.DataFrame({s: sl(r)["trades"] for s, r in res.items()})
    avg = net.mean(axis=1)
    n, nt = len(avg), trades.sum().sum()
    traded = net[trades > 0].stack()
    return {
        "strategy": label, "days": n, "net_yr": avg.mean() * YEAR, "gross_yr": gross.mean(axis=1).mean() * YEAR,
        "sharpe": metrics.sharpe(avg, YEAR), "t": avg.mean() / avg.std() * np.sqrt(n) if avg.std() else 0.0,
        "max_dd": metrics.max_drawdown(avg), "trades_day": nt / (n * len(res)) if n else 0.0,
        "net_bps": net.sum().sum() / nt * 1e4 if nt else 0.0, "gross_bps": gross.sum().sum() / nt * 1e4 if nt else 0.0,
        "win_rate": float((traded > 0).mean()) if len(traded) else float("nan"),
        "sym_pos": float((net.sum() > 0).mean()),
        "psr": metrics.probabilistic_sharpe(avg, 0.0),
        "dsr": metrics.deflated_sharpe(avg, trial_srs) if trial_srs else float("nan"),
        "_avg": avg,
    }


def by_year(avg: pd.Series) -> pd.DataFrame:
    g = avg.groupby(avg.index.year)
    return pd.DataFrame({"return": g.apply(lambda r: (1 + r).prod() - 1), "sharpe": g.apply(lambda r: metrics.sharpe(r, YEAR)),
                         "max_dd": g.apply(metrics.max_drawdown), "sessions": g.size()})


def walk_forward(curves: dict, end: str) -> tuple[pd.Series, list]:
    """Each calendar year, trade the variant with the best net Sharpe over the previous WF_TRAIN_YEARS."""
    frame = pd.DataFrame(curves).loc[:end]
    years = sorted(set(frame.index.year))
    picks, pieces = [], []
    for y in years[WF_TRAIN_YEARS:]:
        train = frame[(frame.index.year >= y - WF_TRAIN_YEARS) & (frame.index.year < y)]
        best = max(train.columns, key=lambda c: metrics.sharpe(train[c], YEAR))
        test = frame.loc[frame.index.year == y, best]
        picks.append((y, best, metrics.sharpe(train[best], YEAR), (1 + test).prod() - 1))
        pieces.append(test)
    return pd.concat(pieces), picks


def parity_report() -> list[str]:
    lines = []
    for s in SYMBOLS:
        try:
            y = yahoo.load(s, "5m", refresh=False)
        except FileNotFoundError:
            continue
        p = alpaca.parity(y, alpaca.load(s, max_age_days=float("inf")))
        if p.empty:
            continue
        same = (p["yahoo_bars"] == p["alpaca_bars"]).mean()
        lines.append(f"  {s:5} {len(p):>3} shared sessions | same bar count {same:>4.0%} | "
                     f"price ratio {p['median_ratio'].median():.4f} | ratio spread p95 median {p['ratio_spread'].median():.5f}, "
                     f"worst {p['ratio_spread'].max():.5f} | first open differs {p['first_open_diff'].abs().median() * 1e4:.1f} bp "
                     f"median, {p['first_open_diff'].abs().max() * 1e4:.1f} worst")
    return lines


def fmt_row(r: dict) -> str:
    return (f"{r['strategy']:38} {r['net_yr']:>7.1%} {r['gross_yr']:>8.1%} {r['sharpe']:>6.2f} {r['t']:>5.1f} "
            f"{r['max_dd']:>7.1%} {r['trades_day']:>5.2f} {r['gross_bps']:>6.1f} >{r['net_bps']:>6.1f} "
            f"{r['win_rate']:>4.0%} {r['sym_pos']:>4.0%} {r['dsr']:>5.2f}")


HEADER = (f"{'strategy':38} {'net/yr':>7} {'gross/yr':>8} {'Sharpe':>6} {'t':>5} {'maxDD':>7} {'trd/d':>5} "
          f"{'bps gross>net':>14} {'win':>4} {'sym+':>4} {'DSR':>5}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cached", action="store_true", help="never download, use the cache as it is")
    a = ap.parse_args()
    sys.stdout = _Tee(sys.stdout, LOG)
    t0 = time.time()

    if not a.cached:
        print(f"Step 1/3: Alpaca 5-minute bars for {len(SYMBOLS)} symbols, {DOWNLOAD_THREADS} at a time "
              "(cached symbols younger than 7 days are skipped)", flush=True)

        def fetch(s):
            t = time.time()
            df = alpaca.load(s, max_age_days=7)
            return s, len(df), df.index.normalize().nunique(), time.time() - t
        with ThreadPoolExecutor(DOWNLOAD_THREADS) as ex:
            for k, fut in enumerate(as_completed([ex.submit(fetch, s) for s in SYMBOLS]), 1):
                s, n, days, secs = fut.result()
                print(f"  [{k:>2}/{len(SYMBOLS)}] {s:5} {n:>7} bars, {days} sessions ({secs:.0f}s)", flush=True)

    vs = list(variants())
    print(f"Step 2/3: running {len(vs)} variants on {len(SYMBOLS)} symbols "
          f"({time.time() - t0:.0f}s so far)", flush=True)
    results = []
    with Pool(min(len(vs), 12), initializer=_init_worker, initargs=(True,)) as pool:
        for k, r in enumerate(pool.imap_unordered(_run_variant, vs), 1):
            results.append(r)
            print(f"  [{k:>2}/{len(vs)}] {r[0]}", flush=True)
    print(f"Step 3/3: results ({time.time() - t0:.0f}s so far)", flush=True)
    leaks = [label for label, leak, _ in results if leak]
    res = {label: r for label, _, r in results}

    flat = None  # the candidate again with the old flat costs, to show what the 1-cent floor changes
    _init_worker(False)
    flat = _run_variant(next(v for v in vs if v[0] == CANDIDATE))[2]

    first = min(r.index[0] for r in res[BASELINE].values()).date()
    last = max(r.index[-1] for r in res[BASELINE].values()).date()
    print(f"\nData: {first} to {last}, {len(next(iter(res[BASELINE].values())))} sessions per symbol")
    print("Parity with the Yahoo 5-minute cache (same bars => ratio spread ~0):")
    print("\n".join(parity_report()) or "  no Yahoo cache to compare")
    if leaks:
        print(f"\nLOOK-AHEAD DETECTED in: {leaks}")

    oos_avg = {l: summarize(l, r, end=OOS_END)["_avg"] for l, r in res.items()}
    trial_srs = [metrics.sharpe(s, 1) for s in oos_avg.values()]
    oos = sorted((summarize(l, r, end=OOS_END, trial_srs=trial_srs) for l, r in res.items()),
                 key=lambda r: -r["sharpe"])
    print(f"\nOUT-OF-SAMPLE {first} to {OOS_END} ({oos[0]['days']} sessions, {len(oos)} variants, DSR across all of them)")
    print(HEADER)
    for r in oos:
        print(fmt_row(r) + ("   <- live paper account" if r["strategy"] == CANDIDATE else ""))

    cand_oos = next(r for r in oos if r["strategy"] == CANDIDATE)
    cand_flat = summarize(CANDIDATE, flat, end=OOS_END)
    sel = summarize(CANDIDATE, res[CANDIDATE], *SELECTION)
    sel_flat = summarize(CANDIDATE, flat, *SELECTION)
    print(f"\nCandidate {CANDIDATE}:")
    print(f"  OOS        net/yr {cand_oos['net_yr']:+.1%}  Sharpe {cand_oos['sharpe']:.2f}  t {cand_oos['t']:.1f}  "
          f"P(Sharpe>0) {cand_oos['psr']:.2f}  max DD {cand_oos['max_dd']:.1%}  win rate {cand_oos['win_rate']:.0%}")
    print(f"  OOS, flat costs (no 1-cent floor): net/yr {cand_flat['net_yr']:+.1%}  Sharpe {cand_flat['sharpe']:.2f}")
    print(f"  Selection window {SELECTION[0]}..{SELECTION[1]} on Alpaca data: net/yr {sel['net_yr']:+.1%} "
          f"(flat costs {sel_flat['net_yr']:+.1%}; Yahoo backtest said +5.4%)")

    cand_years = by_year(summarize(CANDIDATE, res[CANDIDATE])["_avg"])
    base_years = by_year(summarize(BASELINE, res[BASELINE])["_avg"])
    wf, picks = walk_forward({l: summarize(l, r)["_avg"] for l, r in res.items()}, end=str(last))
    wf_years = by_year(wf)
    print(f"\nBy year (net, average of {len(SYMBOLS)} symbols): candidate | buy-open-sell-close | walk-forward pick")
    for y in cand_years.index:
        wf_cell = f"{wf_years.loc[y, 'return']:+7.2%}" if y in wf_years.index else "      -"
        pick = next((p[1] for p in picks if p[0] == y), "")
        print(f"  {y}  {cand_years.loc[y, 'return']:+7.2%} (Sharpe {cand_years.loc[y, 'sharpe']:5.2f}, DD {cand_years.loc[y, 'max_dd']:6.1%})"
              f" | {base_years.loc[y, 'return']:+7.2%} | {wf_cell}  {pick}" + ("   (includes the selection window)" if y == int(SELECTION[0][:4]) else ""))
    print(f"\nWalk-forward overall: net/yr {wf.mean() * YEAR:+.1%}  Sharpe {metrics.sharpe(wf, YEAR):.2f}  "
          f"max DD {metrics.max_drawdown(wf):.1%}  P(Sharpe>0) {metrics.probabilistic_sharpe(wf, 0.0):.2f}")
    write_html(oos, res, cand_years, base_years, wf, picks, first, last)


def write_html(oos, res, cand_years, base_years, wf, picks, first, last) -> None:
    curves = {CANDIDATE + " (live account)": summarize(CANDIDATE, res[CANDIDATE])["_avg"],
              "buy open, sell close": summarize(BASELINE, res[BASELINE])["_avg"],
              "walk-forward pick": wf}
    accts = [Account(name=n, capital=20_000, cash=0,
                     equity=[[d.date().isoformat(), float(v)] for d, v in (20_000 * (1 + c).cumprod()).items()])
             for n, c in curves.items()]
    cls = lambda v: "pos" if v > 0 else "neg" if v < 0 else ""
    oos_rows = "".join(
        f"<tr><td>{escape(r['strategy'])}{' <b>(live)</b>' if r['strategy'] == CANDIDATE else ''}</td>"
        f"<td class='{cls(r['net_yr'])}'>{r['net_yr']:.1%}</td><td>{r['gross_yr']:.1%}</td><td>{r['sharpe']:.2f}</td>"
        f"<td>{r['t']:.1f}</td><td>{r['max_dd']:.1%}</td><td>{r['trades_day']:.2f}</td><td>{r['net_bps']:.1f}</td>"
        f"<td>{r['dsr']:.2f}</td></tr>" for r in oos)
    year_rows = "".join(
        f"<tr><td>{y}</td><td class='{cls(cand_years.loc[y, 'return'])}'>{cand_years.loc[y, 'return']:+.2%}</td>"
        f"<td>{base_years.loc[y, 'return']:+.2%}</td>"
        f"<td>{next((f'{p[3]:+.2%} ({escape(p[1])})' for p in picks if p[0] == y), '-')}</td></tr>"
        for y in cand_years.index)
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>Day-trading 10-year backtest</title><style>{report.CSS}</style></head><body><main>"
            f"<h1>Day-trading backtest, Alpaca 5-minute bars</h1><p class='sub'>{first} to {last} · 12 symbols, flat every "
            f"night · costs 2-4 bps per side plus half a cent per share · out-of-sample = before {SELECTION[0]}</p>"
            f"<div class='card'>{report._chart(accts)}</div>"
            f"<h2>By year (net)</h2><div class='card'><table><tr><th>Year</th><th>Gap fade (live)</th>"
            f"<th>Buy open, sell close</th><th>Walk-forward pick</th></tr>{year_rows}</table></div>"
            f"<h2>All variants, out-of-sample</h2><div class='card'><table><tr><th>Strategy</th><th>Net / yr</th>"
            f"<th>Gross / yr</th><th>Sharpe</th><th>t</th><th>Max DD</th><th>Trades / day</th><th>Net bps / trade</th>"
            f"<th>DSR</th></tr>{oos_rows}</table></div></main></body></html>")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "longrun.html").write_text(page, encoding="utf-8")
    print(f"\nReport: {OUT / 'longrun.html'}")


if __name__ == "__main__":
    main()
