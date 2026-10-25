"""Self-contained HTML dashboard for paper accounts (no external files, light and dark mode)."""
from html import escape

import pandas as pd

from .engine import Account

COLORS = ["#2563eb", "#d97706", "#059669", "#db2777", "#7c3aed", "#0891b2"]

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--fg:#1c1c1a;--muted:#6b6b66;--line:#e4e4df;--pos:#15803d;--neg:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--fg:#ececea;--muted:#9a9a94;
--line:#33332f;--pos:#4ade80;--neg:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 10px}.sub{color:var(--muted);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500;font-size:13px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;
font-size:13px}.legend i{display:inline-block;width:12px;height:3px;margin-right:6px;vertical-align:middle}
svg text{fill:var(--muted);font-size:11px}svg .grid{stroke:var(--line)}
"""


def _pct(x: float) -> str:
    cls = "pos" if x > 0 else "neg" if x < 0 else ""
    return f'<span class="{cls}">{x:+.2%}</span>'


def _eur(x: float) -> str:
    return f"€{x:,.0f}"


def _chart(accounts: list[Account], w: int = 960, h: int = 300, pad: int = 44) -> str:
    curves = {a.name: pd.Series({pd.Timestamp(d): v / a.capital * 100 for d, v in a.equity}) for a in accounts}
    lo = min(s.min() for s in curves.values())
    hi = max(s.max() for s in curves.values())
    lo, hi = (lo - 1, hi + 1) if hi - lo < 2 else (lo, hi)
    t0 = min(s.index[0] for s in curves.values())
    t1 = max(s.index[-1] for s in curves.values())
    span = max((t1 - t0).days, 1)
    x = lambda t: pad + (t - t0).days / span * (w - pad - 8)
    y = lambda v: 8 + (hi - v) / (hi - lo) * (h - pad)
    parts = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        parts.append(f'<line class="grid" x1="{pad}" x2="{w - 8}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                     f'<text x="{pad - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    parts.append(f'<text x="{pad}" y="{h - 12}">{t0.date()}</text>'
                 f'<text x="{w - 8}" y="{h - 12}" text-anchor="end">{t1.date()}</text>')
    for color, (name, s) in zip(COLORS, curves.items()):
        pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in s.items())
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
    legend = "".join(f'<span><i style="background:{c}"></i>{escape(a.name)}</span>' for c, a in zip(COLORS, accounts))
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Equity curves, start = 100">'
            f'{"".join(parts)}</svg><div class="legend">{legend}</div>')


def _max_dd(a: Account) -> float:
    eq = pd.Series([v for _, v in a.equity])
    return float((eq / eq.cummax() - 1).min())


def render(accounts: list[Account], title: str, subtitle: str, benchmark: str) -> str:
    bench = next((a for a in accounts if a.name == benchmark), None)
    bench_ret = bench.value / bench.capital - 1 if bench else 0.0
    rows = []
    for a in accounts:
        ret = a.value / a.capital - 1
        vs = "" if a is bench else _pct(ret - bench_ret)
        rows.append(f"<tr><td>{escape(a.name)}</td><td>{_eur(a.value)}</td><td>{_pct(ret)}</td><td>{vs}</td>"
                    f"<td>{_pct(_max_dd(a))}</td><td>{_eur(a.fees)}</td><td>{len(a.trades)}</td></tr>")
    holdings = []
    for a in accounts:
        pos = ", ".join(f"{escape(t)} {_eur(v)}" for t, v in sorted(a.positions.items(), key=lambda kv: -kv[1]))
        holdings.append(f"<tr><td>{escape(a.name)}</td><td style='text-align:left;white-space:normal'>"
                        f"{pos or 'cash'}</td><td>{_eur(a.cash)}</td></tr>")
    recent = sorted(((t, a.name) for a in accounts for t in a.trades), key=lambda x: x[0]["date"], reverse=True)[:30]
    trades = "".join(f"<tr><td>{t['date']}</td><td>{escape(n)}</td><td>{t['side']}</td><td>{escape(t['ticker'])}</td>"
                     f"<td>{_eur(t['eur'])}</td><td>€{t['cost']:.2f}</td></tr>" for t, n in recent)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>{CSS}</style></head><body><main>
<h1>{escape(title)}</h1><p class="sub">{escape(subtitle)}</p>
<div class="card">{_chart(accounts)}</div>
<h2>Accounts</h2><div class="card"><table><tr><th>Strategy</th><th>Value</th><th>Return</th>
<th>vs {escape(benchmark)}</th><th>Max drawdown</th><th>Costs paid</th><th>Trades</th></tr>{"".join(rows)}</table></div>
<h2>Holdings</h2><div class="card"><table><tr><th>Strategy</th><th style="text-align:left">Positions</th>
<th>Cash</th></tr>{"".join(holdings)}</table></div>
<h2>Recent trades</h2><div class="card"><table><tr><th>Filled</th><th>Strategy</th><th>Side</th><th>Ticker</th>
<th>Amount</th><th>Cost</th></tr>{trades or '<tr><td colspan="6">No trades yet</td></tr>'}</table></div>
</main></body></html>"""
