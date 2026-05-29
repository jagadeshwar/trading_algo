"""Reusable Plotly chart components."""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick_chart(
    ohlcv: pd.DataFrame,
    signals_df: pd.DataFrame | None = None,
    symbol: str = "",
    interval_min: int = 5,
    show_emas: list[int] | None = None,
    show_volume: bool = True,
    show_vix: pd.Series | None = None,
) -> go.Figure:
    """Full candlestick chart with EMA overlays, volume, optional VIX."""
    rows = 2 if show_volume else 1
    if show_vix is not None:
        rows += 1

    row_heights = {1: [0.7, 0.3], 2: [0.55, 0.25, 0.2]}.get(rows - 1, [1.0])

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=[None] * rows,
    )

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["open"], high=ohlcv["high"],
        low=ohlcv["low"],  close=ohlcv["close"],
        name="OHLCV",
        increasing_line_color="#00c896",
        decreasing_line_color="#ff4c6a",
        increasing_fillcolor="#00c896",
        decreasing_fillcolor="#ff4c6a",
    ), row=1, col=1)

    # ── EMAs ──────────────────────────────────────────────────────────────────
    ema_colors = {9: "#ffd166", 21: "#06d6a0", 50: "#118ab2", 200: "#ef476f"}
    for period in (show_emas or [9, 21]):
        if period in ema_colors:
            ema = ohlcv["close"].ewm(span=period, adjust=False).mean()
            fig.add_trace(go.Scatter(
                x=ohlcv.index, y=ema, name=f"EMA{period}",
                line=dict(color=ema_colors[period], width=1.5),
                opacity=0.85,
            ), row=1, col=1)

    # ── Signal markers ────────────────────────────────────────────────────────
    if signals_df is not None and "direction" in signals_df.columns:
        long_idx  = signals_df.index[signals_df["direction"] == 1]
        short_idx = signals_df.index[signals_df["direction"] == -1]

        if len(long_idx):
            fig.add_trace(go.Scatter(
                x=long_idx,
                y=ohlcv.loc[long_idx, "low"] * 0.999 if len(long_idx) else [],
                mode="markers",
                name="Long Entry",
                marker=dict(symbol="triangle-up", size=12, color="#00c896"),
            ), row=1, col=1)

        if len(short_idx):
            fig.add_trace(go.Scatter(
                x=short_idx,
                y=ohlcv.loc[short_idx, "high"] * 1.001 if len(short_idx) else [],
                mode="markers",
                name="Short Entry",
                marker=dict(symbol="triangle-down", size=12, color="#ff4c6a"),
            ), row=1, col=1)

    # ── Volume ────────────────────────────────────────────────────────────────
    if show_volume:
        colors = ["#00c896" if c >= o else "#ff4c6a"
                  for c, o in zip(ohlcv["close"], ohlcv["open"])]
        fig.add_trace(go.Bar(
            x=ohlcv.index, y=ohlcv["volume"],
            name="Volume", marker_color=colors, opacity=0.6,
        ), row=2, col=1)

    # ── VIX ───────────────────────────────────────────────────────────────────
    if show_vix is not None:
        vix_row = 3 if show_volume else 2
        fig.add_trace(go.Scatter(
            x=show_vix.index, y=show_vix.values,
            name="India VIX", line=dict(color="#a78bfa", width=1.5),
        ), row=vix_row, col=1)
        for level, color in [(20, "rgba(255,209,102,0.3)"), (25, "rgba(255,76,106,0.3)")]:
            fig.add_hline(y=level, line_dash="dash", line_color=color,
                         row=vix_row, col=1, annotation_text=f"VIX {level}")

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=f"{symbol} — {interval_min}min", font=dict(size=16)),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e2130")
    fig.update_yaxes(showgrid=True, gridcolor="#1e2130")
    return fig


def equity_curve(pnl_series: pd.Series, title: str = "Equity Curve") -> go.Figure:
    cum = (1 + pnl_series).cumprod() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values * 100,
        fill="tozeroy",
        fillcolor="rgba(0,200,150,0.15)",
        line=dict(color="#00c896", width=2),
        name="Return %",
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        yaxis_title="Cumulative Return (%)",
        height=300, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
