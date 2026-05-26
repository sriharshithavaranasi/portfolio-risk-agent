"""
Report generator — turns a RiskMemo + raw portfolio data into polished output.

Architecture
------------
Three clear layers:

  1. DATA ASSEMBLY  — build_report() pulls from RiskMemo, holdings, and prices
                      to populate a PortfolioReport dataclass. No formatting here.

  2. MARKDOWN       — render_markdown() turns PortfolioReport into a .md string.
                      Clean for terminal output, GitHub, Slack (with renderer).

  3. HTML EMAIL     — render_html() turns PortfolioReport into a self-contained
                      HTML email. All CSS is inline — required for email clients
                      that strip <style> blocks (Gmail, Outlook, Apple Mail).

Sections
--------
  Executive Summary   — key metrics at a glance + analyst narrative
  Top Movers          — best and worst performing holdings yesterday
  Risk Flags          — all warnings collected in one place
  Portfolio Concentration — weight table, HHI, overweight flags
  Macro Context       — benchmark performance and market regime
  Watchlist           — positions that need monitoring today

Public API
----------
  build_report(memo, holdings, prices)  -> PortfolioReport
  render_markdown(report)               -> str
  render_html(report)                   -> str
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from html import escape
from typing import Literal

import pandas as pd

from agent.memo import RiskMemo
from agent.risk import compute_portfolio_weights
from config import BENCHMARK_TICKER, CONCENTRATION_THRESHOLD


# ---------------------------------------------------------------------------
# Section data models
# ---------------------------------------------------------------------------

@dataclass
class TopMover:
    ticker:               str
    shares:               float
    daily_change_pct:     float   # e.g. 0.032 = +3.2%
    daily_change_dollars: float   # dollar PnL for this position
    direction:            Literal["up", "down", "flat"]


@dataclass
class ConcentrationRow:
    ticker: str
    weight: float          # e.g. 0.42 = 42%
    market_value: float
    flagged: bool          # True if weight > CONCENTRATION_THRESHOLD


@dataclass
class WatchlistItem:
    ticker: str
    weight: float
    reasons: list[str]     # e.g. ["Overweight (42%)", "Negative news"]


@dataclass
class MacroContext:
    benchmark_ticker:      str
    benchmark_daily_pct:   float   # SPY yesterday's move
    benchmark_ytd_pct:     float   # SPY year-to-date
    regime:                str     # "Bull", "Bear", "Sideways"
    regime_note:           str     # one-line explanation


@dataclass
class PortfolioReport:
    """All sections of the morning brief, ready to render in any format."""
    generated_on:       str
    risk_level:         str
    executive_summary:  dict         # flat key-value metrics + narrative
    top_movers:         list[TopMover]
    risk_flags:         list[str]
    concentration:      list[ConcentrationRow]
    macro_context:      MacroContext
    watchlist:          list[WatchlistItem]
    recommendations:    list[str]
    news_highlights:    list[dict]   # passed through from RiskMemo


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_report(
    memo: RiskMemo,
    holdings: list[dict],
    prices: dict[str, pd.DataFrame],
) -> PortfolioReport:
    """
    Assemble all report sections from the RiskMemo and raw price data.

    Args:
        memo:     Structured output from generate_memo() in memo.py.
        holdings: List of holding dicts from loader.py.
        prices:   Price DataFrames from prices.py (includes benchmark).

    Returns:
        PortfolioReport ready to pass to render_markdown() or render_html().
    """
    weights      = compute_portfolio_weights(holdings, prices)
    total_value  = _portfolio_value(holdings, prices)

    return PortfolioReport(
        generated_on=      date.today().isoformat(),
        risk_level=        memo.overall_risk_level,
        executive_summary= _build_executive_summary(memo, total_value),
        top_movers=        _build_top_movers(holdings, prices),
        risk_flags=        _all_risk_flags(memo),
        concentration=     _build_concentration(holdings, prices, weights),
        macro_context=     _build_macro_context(prices),
        watchlist=         _build_watchlist(holdings, prices, weights, memo),
        recommendations=   memo.recommendations,
        news_highlights=   memo.news_highlights,
    )


def _portfolio_value(holdings: list[dict], prices: dict[str, pd.DataFrame]) -> float:
    total = 0.0
    for h in holdings:
        ticker = h["ticker"]
        if ticker in prices:
            total += float(prices[ticker]["close"].iloc[-1]) * h["shares"]
    return total


def _build_executive_summary(memo: RiskMemo, total_value: float) -> dict:
    return {
        "total_value":       total_value,
        "daily_pnl_dollars": memo.daily_pnl_dollars,
        "daily_pnl_pct":     memo.daily_pnl_pct,
        "portfolio_beta":    memo.portfolio_beta,
        "max_drawdown_pct":  memo.max_drawdown_pct,
        "hhi_score":         memo.hhi_score,
        "largest_position":  memo.largest_position,
        "narrative":         memo.analyst_commentary,
    }


def _build_top_movers(
    holdings: list[dict],
    prices: dict[str, pd.DataFrame],
) -> list[TopMover]:
    movers = []
    for h in holdings:
        ticker = h["ticker"]
        if ticker not in prices or len(prices[ticker]) < 2:
            continue
        closes     = prices[ticker]["close"]
        prev_close = float(closes.iloc[-2])
        last_close = float(closes.iloc[-1])
        chg_pct    = (last_close - prev_close) / prev_close if prev_close else 0
        chg_dollar = (last_close - prev_close) * h["shares"]
        movers.append(TopMover(
            ticker=               ticker,
            shares=               h["shares"],
            daily_change_pct=     chg_pct,
            daily_change_dollars= chg_dollar,
            direction=            "up" if chg_pct > 0 else ("down" if chg_pct < 0 else "flat"),
        ))
    # Sort: biggest absolute move first
    movers.sort(key=lambda m: abs(m.daily_change_pct), reverse=True)
    return movers


def _all_risk_flags(memo: RiskMemo) -> list[str]:
    flags = list(memo.risk_flags)
    for ticker in memo.concentration_flags:
        flag = f"{ticker} exceeds {CONCENTRATION_THRESHOLD:.0%} concentration threshold"
        if flag not in flags:
            flags.append(flag)
    return flags


def _build_concentration(
    holdings: list[dict],
    prices: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> list[ConcentrationRow]:
    rows = []
    for h in holdings:
        ticker = h["ticker"]
        if ticker not in prices:
            continue
        mv = float(prices[ticker]["close"].iloc[-1]) * h["shares"]
        rows.append(ConcentrationRow(
            ticker=       ticker,
            weight=       weights.get(ticker, 0.0),
            market_value= mv,
            flagged=      weights.get(ticker, 0.0) > CONCENTRATION_THRESHOLD,
        ))
    rows.sort(key=lambda r: r.weight, reverse=True)
    return rows


def _build_macro_context(prices: dict[str, pd.DataFrame]) -> MacroContext:
    benchmark = prices.get(BENCHMARK_TICKER)

    if benchmark is None or len(benchmark) < 2:
        return MacroContext(
            benchmark_ticker=    BENCHMARK_TICKER,
            benchmark_daily_pct= 0.0,
            benchmark_ytd_pct=   0.0,
            regime=              "Unknown",
            regime_note=         "Insufficient benchmark data.",
        )

    closes            = benchmark["close"]
    daily_pct         = float((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2])
    ytd_pct           = float((closes.iloc[-1] - closes.iloc[0])  / closes.iloc[0])

    # Simple regime heuristic: compare the latest close to the 50-day simple moving average.
    # Above SMA-50 → Bull; below by >5% → Bear; otherwise → Sideways.
    sma50   = float(closes.tail(50).mean())
    last    = float(closes.iloc[-1])
    gap_pct = (last - sma50) / sma50

    if gap_pct > 0.02:
        regime, note = "Bull", f"{BENCHMARK_TICKER} is {gap_pct:.1%} above its 50-day average."
    elif gap_pct < -0.05:
        regime, note = "Bear", f"{BENCHMARK_TICKER} is {abs(gap_pct):.1%} below its 50-day average."
    else:
        regime, note = "Sideways", f"{BENCHMARK_TICKER} is near its 50-day average ({gap_pct:+.1%})."

    return MacroContext(
        benchmark_ticker=    BENCHMARK_TICKER,
        benchmark_daily_pct= daily_pct,
        benchmark_ytd_pct=   ytd_pct,
        regime=              regime,
        regime_note=         note,
    )


def _build_watchlist(
    holdings: list[dict],
    prices: dict[str, pd.DataFrame],
    weights: dict[str, float],
    memo: RiskMemo,
) -> list[WatchlistItem]:
    """
    Flag positions that need monitoring today.

    A holding makes the watchlist if it has at least one of:
      - Overweight (above CONCENTRATION_THRESHOLD)
      - Appeared in a news highlight
      - Had the largest single-day loss
    """
    news_tickers  = {h["ticker"] for h in memo.news_highlights}
    movers        = _build_top_movers(holdings, prices)
    worst_mover   = movers[-1].ticker if movers else None

    items: dict[str, WatchlistItem] = {}

    for h in holdings:
        ticker  = h["ticker"]
        weight  = weights.get(ticker, 0.0)
        reasons: list[str] = []

        if weight > CONCENTRATION_THRESHOLD:
            reasons.append(f"Overweight at {weight:.1%} (threshold {CONCENTRATION_THRESHOLD:.0%})")
        if ticker in news_tickers:
            reasons.append("Active in news — review headlines")
        if ticker == worst_mover:
            reasons.append("Largest daily loss in portfolio")

        if reasons:
            items[ticker] = WatchlistItem(ticker=ticker, weight=weight, reasons=reasons)

    return sorted(items.values(), key=lambda i: i.weight, reverse=True)


# ---------------------------------------------------------------------------
# Risk level styling helpers (shared by both renderers)
# ---------------------------------------------------------------------------

_RISK_COLORS = {
    "LOW":      {"bg": "#d4edda", "text": "#155724", "badge": "#28a745"},
    "MEDIUM":   {"bg": "#fff3cd", "text": "#856404", "badge": "#ffc107"},
    "HIGH":     {"bg": "#f8d7da", "text": "#721c24", "badge": "#dc3545"},
    "CRITICAL": {"bg": "#f5c6cb", "text": "#491217", "badge": "#b21f2d"},
}

def _risk_colors(level: str) -> dict:
    return _RISK_COLORS.get(level.upper(), _RISK_COLORS["MEDIUM"])


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_markdown(report: PortfolioReport) -> str:
    """Render a PortfolioReport as a GitHub-flavoured Markdown string."""
    parts: list[str] = []
    s = report.executive_summary

    # Header
    parts.append(f"# Morning Portfolio Risk Brief — {report.generated_on}")
    parts.append(f"\n**Risk Level: {report.risk_level}**\n")

    # Executive Summary
    parts.append("## Executive Summary\n")
    pnl_sign = "+" if s["daily_pnl_dollars"] >= 0 else ""
    parts.append(f"| Metric | Value |")
    parts.append(f"|--------|-------|")
    parts.append(f"| Portfolio Value | ${s['total_value']:,.2f} |")
    parts.append(f"| Daily PnL | {pnl_sign}${s['daily_pnl_dollars']:,.2f} ({pnl_sign}{s['daily_pnl_pct']*100:.2f}%) |")
    parts.append(f"| Portfolio Beta | {s['portfolio_beta']:.2f} |")
    parts.append(f"| Max Drawdown | {s['max_drawdown_pct']*100:.1f}% |")
    parts.append(f"| HHI Score | {s['hhi_score']:.3f} |")
    parts.append(f"| Largest Position | {s['largest_position']} |")
    parts.append(f"\n{s['narrative']}\n")

    # Top Movers
    parts.append("## Top Movers\n")
    parts.append("| Ticker | Change % | P&L ($) | Direction |")
    parts.append("|--------|----------|---------|-----------|")
    for m in report.top_movers:
        sign   = "+" if m.daily_change_pct >= 0 else ""
        arrow  = "▲" if m.direction == "up" else ("▼" if m.direction == "down" else "—")
        parts.append(
            f"| {m.ticker} | {sign}{m.daily_change_pct*100:.2f}% "
            f"| {sign}${m.daily_change_dollars:,.2f} | {arrow} |"
        )
    parts.append("")

    # Risk Flags
    parts.append("## Risk Flags\n")
    if report.risk_flags:
        for flag in report.risk_flags:
            parts.append(f"- ⚠️  {flag}")
    else:
        parts.append("_No risk flags today._")
    parts.append("")

    # Portfolio Concentration
    parts.append("## Portfolio Concentration\n")
    parts.append(f"**HHI Score:** {report.executive_summary['hhi_score']:.3f} "
                 f"(0 = perfectly diversified, 1 = fully concentrated)\n")
    parts.append("| Ticker | Weight | Market Value | Status |")
    parts.append("|--------|--------|--------------|--------|")
    for row in report.concentration:
        status = "⚠️ Overweight" if row.flagged else "OK"
        parts.append(
            f"| {row.ticker} | {row.weight*100:.1f}% "
            f"| ${row.market_value:,.2f} | {status} |"
        )
    parts.append("")

    # Macro Context
    mc   = report.macro_context
    sign = "+" if mc.benchmark_daily_pct >= 0 else ""
    parts.append("## Macro Context\n")
    parts.append(f"**Benchmark:** {mc.benchmark_ticker}  ")
    parts.append(f"**Regime:** {mc.regime}  ")
    parts.append(f"**Daily Move:** {sign}{mc.benchmark_daily_pct*100:.2f}%  ")
    parts.append(f"**YTD:** {sign}{mc.benchmark_ytd_pct*100:.2f}%  ")
    parts.append(f"\n_{mc.regime_note}_\n")

    # News Highlights
    if report.news_highlights:
        parts.append("## News Highlights\n")
        for item in report.news_highlights:
            parts.append(f"**{item['ticker']}** — {item['digest']}\n")

    # Watchlist
    parts.append("## Watchlist\n")
    if report.watchlist:
        for item in report.watchlist:
            reasons = "; ".join(item.reasons)
            parts.append(f"- **{item.ticker}** ({item.weight*100:.1f}%): {reasons}")
    else:
        parts.append("_No positions flagged for watchlist today._")
    parts.append("")

    # Recommendations
    parts.append("## Recommendations\n")
    for rec in report.recommendations:
        parts.append(f"1. {rec}")
    parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML email renderer
# ---------------------------------------------------------------------------
# All CSS is inline. Email clients (Gmail, Outlook, Apple Mail) strip <style>
# blocks, so every element must carry its own style attribute.

def render_html(report: PortfolioReport) -> str:
    """Render a PortfolioReport as a self-contained HTML email string."""
    colors = _risk_colors(report.risk_level)
    s      = report.executive_summary

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Portfolio Risk Brief — {escape(report.generated_on)}</title>
</head>
<body style="margin:0;padding:0;background:#f4f4f4;font-family:Arial,Helvetica,sans-serif;">

<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f4;padding:24px 0;">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:6px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">

  <!-- HEADER -->
  <tr>
    <td style="background:{colors['badge']};padding:24px 32px;">
      <p style="margin:0;color:#ffffff;font-size:12px;text-transform:uppercase;letter-spacing:1px;">Morning Risk Brief</p>
      <h1 style="margin:4px 0 0;color:#ffffff;font-size:22px;">Portfolio Risk Report</h1>
      <p style="margin:6px 0 0;color:rgba(255,255,255,0.85);font-size:13px;">{escape(report.generated_on)}</p>
    </td>
  </tr>

  <!-- RISK BADGE -->
  <tr>
    <td style="padding:20px 32px 0;">
      <span style="display:inline-block;background:{colors['bg']};color:{colors['text']};
                   border:1px solid {colors['badge']};border-radius:4px;
                   padding:6px 14px;font-size:13px;font-weight:bold;">
        RISK LEVEL: {escape(report.risk_level)}
      </span>
    </td>
  </tr>

  <!-- EXECUTIVE SUMMARY -->
  {_html_section("Executive Summary")}
  <tr><td style="padding:0 32px 8px;">
    <table width="100%" cellpadding="8" cellspacing="0" style="border-collapse:collapse;font-size:14px;">
      {_html_metric_row("Portfolio Value",   f"${s['total_value']:,.2f}")}
      {_html_pnl_row(s['daily_pnl_dollars'], s['daily_pnl_pct'])}
      {_html_metric_row("Portfolio Beta",    f"{s['portfolio_beta']:.2f}")}
      {_html_metric_row("Max Drawdown",      f"{s['max_drawdown_pct']*100:.1f}%")}
      {_html_metric_row("HHI Score",         f"{s['hhi_score']:.3f}")}
      {_html_metric_row("Largest Position",  escape(s['largest_position']))}
    </table>
    <p style="font-size:14px;color:#333;line-height:1.7;margin:16px 0 0;">
      {escape(s['narrative']).replace(chr(10), '<br>')}
    </p>
  </td></tr>

  <!-- TOP MOVERS -->
  {_html_section("Top Movers")}
  <tr><td style="padding:0 32px 8px;">
    <table width="100%" cellpadding="8" cellspacing="0"
           style="border-collapse:collapse;font-size:14px;">
      <tr style="background:#f8f9fa;">
        <th style="text-align:left;padding:8px;border-bottom:2px solid #dee2e6;">Ticker</th>
        <th style="text-align:right;padding:8px;border-bottom:2px solid #dee2e6;">Change %</th>
        <th style="text-align:right;padding:8px;border-bottom:2px solid #dee2e6;">P&amp;L ($)</th>
      </tr>
      {"".join(_html_mover_row(m) for m in report.top_movers)}
    </table>
  </td></tr>

  <!-- RISK FLAGS -->
  {_html_section("Risk Flags")}
  <tr><td style="padding:0 32px 16px;">
    {"".join(_html_flag(f) for f in report.risk_flags)
      if report.risk_flags
      else '<p style="font-size:14px;color:#666;font-style:italic;">No risk flags today.</p>'}
  </td></tr>

  <!-- PORTFOLIO CONCENTRATION -->
  {_html_section("Portfolio Concentration")}
  <tr><td style="padding:0 32px 8px;">
    <p style="font-size:13px;color:#555;margin:0 0 12px;">
      HHI Score: <strong>{s['hhi_score']:.3f}</strong>
      &nbsp;(0 = perfectly diversified · 1 = fully concentrated)
    </p>
    <table width="100%" cellpadding="8" cellspacing="0"
           style="border-collapse:collapse;font-size:14px;">
      <tr style="background:#f8f9fa;">
        <th style="text-align:left;padding:8px;border-bottom:2px solid #dee2e6;">Ticker</th>
        <th style="text-align:right;padding:8px;border-bottom:2px solid #dee2e6;">Weight</th>
        <th style="text-align:right;padding:8px;border-bottom:2px solid #dee2e6;">Market Value</th>
        <th style="text-align:center;padding:8px;border-bottom:2px solid #dee2e6;">Status</th>
      </tr>
      {"".join(_html_concentration_row(r) for r in report.concentration)}
    </table>
  </td></tr>

  <!-- MACRO CONTEXT -->
  {_html_section("Macro Context")}
  <tr><td style="padding:0 32px 16px;">
    {_html_macro(report.macro_context)}
  </td></tr>

  <!-- NEWS HIGHLIGHTS -->
  {_html_section("News Highlights") if report.news_highlights else ""}
  {"".join(_html_news(n) for n in report.news_highlights)}

  <!-- WATCHLIST -->
  {_html_section("Watchlist")}
  <tr><td style="padding:0 32px 16px;">
    {"".join(_html_watchlist_item(w) for w in report.watchlist)
      if report.watchlist
      else '<p style="font-size:14px;color:#666;font-style:italic;">No positions flagged today.</p>'}
  </td></tr>

  <!-- RECOMMENDATIONS -->
  {_html_section("Recommendations")}
  <tr><td style="padding:0 32px 24px;">
    <ol style="font-size:14px;color:#333;line-height:1.8;margin:0;padding-left:20px;">
      {"".join(f"<li>{escape(r)}</li>" for r in report.recommendations)}
    </ol>
  </td></tr>

  <!-- FOOTER -->
  <tr>
    <td style="background:#f8f9fa;padding:16px 32px;border-top:1px solid #dee2e6;">
      <p style="margin:0;font-size:11px;color:#999;text-align:center;">
        Generated by Portfolio Risk Agent · {escape(report.generated_on)}
        · For informational purposes only. Not financial advice.
      </p>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTML helper functions
# ---------------------------------------------------------------------------

def _html_section(title: str) -> str:
    return (
        f'<tr><td style="padding:24px 32px 8px;">'
        f'<h2 style="margin:0;font-size:16px;color:#212529;'
        f'border-bottom:2px solid #dee2e6;padding-bottom:8px;">'
        f'{escape(title)}</h2></td></tr>'
    )


def _html_metric_row(label: str, value: str) -> str:
    return (
        f'<tr style="border-bottom:1px solid #f0f0f0;">'
        f'<td style="padding:8px;color:#555;">{escape(label)}</td>'
        f'<td style="padding:8px;text-align:right;font-weight:bold;color:#212529;">'
        f'{value}</td></tr>'
    )


def _html_pnl_row(dollars: float, pct: float) -> str:
    sign  = "+" if dollars >= 0 else ""
    color = "#28a745" if dollars >= 0 else "#dc3545"
    value = (
        f'<span style="color:{color};font-weight:bold;">'
        f'{sign}${dollars:,.2f} ({sign}{pct*100:.2f}%)</span>'
    )
    return (
        f'<tr style="border-bottom:1px solid #f0f0f0;">'
        f'<td style="padding:8px;color:#555;">Daily P&amp;L</td>'
        f'<td style="padding:8px;text-align:right;">{value}</td></tr>'
    )


def _html_mover_row(m: TopMover) -> str:
    color = "#28a745" if m.direction == "up" else ("#dc3545" if m.direction == "down" else "#555")
    sign  = "+" if m.daily_change_pct >= 0 else ""
    arrow = "▲" if m.direction == "up" else ("▼" if m.direction == "down" else "—")
    return (
        f'<tr style="border-bottom:1px solid #f0f0f0;">'
        f'<td style="padding:8px;font-weight:bold;">{escape(m.ticker)}</td>'
        f'<td style="padding:8px;text-align:right;color:{color};">'
        f'{arrow} {sign}{m.daily_change_pct*100:.2f}%</td>'
        f'<td style="padding:8px;text-align:right;color:{color};">'
        f'{sign}${m.daily_change_dollars:,.2f}</td></tr>'
    )


def _html_flag(flag: str) -> str:
    return (
        f'<div style="display:flex;align-items:flex-start;margin-bottom:8px;">'
        f'<span style="color:#dc3545;font-size:16px;margin-right:8px;flex-shrink:0;">⚠</span>'
        f'<span style="font-size:14px;color:#333;">{escape(flag)}</span></div>'
    )


def _html_concentration_row(r: ConcentrationRow) -> str:
    bg     = "#fff3cd" if r.flagged else "transparent"
    status = ('<span style="color:#856404;font-weight:bold;">⚠ Overweight</span>'
              if r.flagged else
              '<span style="color:#28a745;">✓ OK</span>')
    return (
        f'<tr style="background:{bg};border-bottom:1px solid #f0f0f0;">'
        f'<td style="padding:8px;font-weight:bold;">{escape(r.ticker)}</td>'
        f'<td style="padding:8px;text-align:right;">{r.weight*100:.1f}%</td>'
        f'<td style="padding:8px;text-align:right;">${r.market_value:,.2f}</td>'
        f'<td style="padding:8px;text-align:center;">{status}</td></tr>'
    )


def _html_macro(mc: MacroContext) -> str:
    sign        = "+" if mc.benchmark_daily_pct >= 0 else ""
    ytd_sign    = "+" if mc.benchmark_ytd_pct >= 0 else ""
    regime_colors = {"Bull": "#28a745", "Bear": "#dc3545", "Sideways": "#ffc107"}
    regime_color  = regime_colors.get(mc.regime, "#555")
    return (
        f'<table cellpadding="0" cellspacing="0" style="font-size:14px;">'
        f'<tr><td style="color:#555;padding:4px 16px 4px 0;">Benchmark</td>'
        f'<td style="font-weight:bold;">{escape(mc.benchmark_ticker)}</td></tr>'
        f'<tr><td style="color:#555;padding:4px 16px 4px 0;">Regime</td>'
        f'<td style="font-weight:bold;color:{regime_color};">{escape(mc.regime)}</td></tr>'
        f'<tr><td style="color:#555;padding:4px 16px 4px 0;">Daily Move</td>'
        f'<td style="font-weight:bold;">{sign}{mc.benchmark_daily_pct*100:.2f}%</td></tr>'
        f'<tr><td style="color:#555;padding:4px 16px 4px 0;">YTD Return</td>'
        f'<td style="font-weight:bold;">{ytd_sign}{mc.benchmark_ytd_pct*100:.2f}%</td></tr>'
        f'</table>'
        f'<p style="font-size:13px;color:#666;font-style:italic;margin:10px 0 0;">'
        f'{escape(mc.regime_note)}</p>'
    )


def _html_news(item: dict) -> str:
    return (
        f'<tr><td style="padding:4px 32px 4px;">'
        f'<p style="margin:0;font-size:14px;">'
        f'<strong style="color:#212529;">{escape(item.get("ticker",""))}</strong>'
        f' &mdash; {escape(item.get("digest",""))}</p></td></tr>'
    )


def _html_watchlist_item(w: WatchlistItem) -> str:
    reasons_html = "".join(
        f'<li style="margin-bottom:2px;">{escape(r)}</li>' for r in w.reasons
    )
    return (
        f'<div style="margin-bottom:14px;padding:12px;background:#f8f9fa;'
        f'border-left:3px solid #ffc107;border-radius:4px;">'
        f'<p style="margin:0 0 4px;font-size:14px;font-weight:bold;">'
        f'{escape(w.ticker)} '
        f'<span style="font-weight:normal;color:#555;">({w.weight*100:.1f}%)</span></p>'
        f'<ul style="margin:0;padding-left:16px;font-size:13px;color:#555;">'
        f'{reasons_html}</ul></div>'
    )
