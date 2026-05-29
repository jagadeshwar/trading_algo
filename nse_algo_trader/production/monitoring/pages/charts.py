"""Charts page — interactive candlestick with indicators and signals."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st


SYMBOLS = [
    "NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX",
    "NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ", "NSE:INFY-EQ",
    "NSE:ICICIBANK-EQ", "NSE:SBIN-EQ", "NSE:AXISBANK-EQ",
    "NSE:KOTAKBANK-EQ", "NSE:LT-EQ", "NSE:ITC-EQ",
]
INTERVALS = {1: "1 min", 5: "5 min", 15: "15 min", 1440: "Daily"}


def _load_parquet(symbol: str, interval_min: int) -> pd.DataFrame | None:
    slug = symbol.replace(":", "_").replace("-", "_")
    path = Path(f"data/ohlcv/{slug}_{interval_min}min.parquet")
    if not path.exists():
        return None
    import pyarrow.parquet as pq
    df = pq.read_table(path).to_pandas()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
    return df.sort_index()


def _load_features(symbol: str, interval_min: int) -> pd.DataFrame | None:
    slug = symbol.replace(":", "_").replace("-", "_")
    path = Path(f"data/features/{slug}_{interval_min}min_features.parquet")
    if not path.exists():
        return None
    import pyarrow.parquet as pq
    df = pq.read_table(path).to_pandas()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
    return df.sort_index()


def render():
    st.title("📈 Charts")

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
    with c1:
        symbol = st.selectbox("Symbol", SYMBOLS,
                              index=SYMBOLS.index("NSE:HDFCBANK-EQ"))
    with c2:
        interval_min = st.selectbox("Timeframe", list(INTERVALS.keys()),
                                    format_func=lambda x: INTERVALS[x], index=1)
    with c3:
        bars = st.selectbox("Show last", [100, 250, 500, 1000, 2000],
                            index=1, format_func=lambda x: f"{x} bars")
    with c4:
        show_signals = st.toggle("Signals", value=True)

    # ── Indicator toggles ─────────────────────────────────────────────────────
    ec1, ec2, ec3, ec4, ec5 = st.columns(5)
    ema_9   = ec1.toggle("EMA 9",   value=True)
    ema_21  = ec2.toggle("EMA 21",  value=True)
    ema_50  = ec3.toggle("EMA 50",  value=False)
    ema_200 = ec4.toggle("EMA 200", value=False)
    show_vol= ec5.toggle("Volume",  value=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    ohlcv = _load_parquet(symbol, interval_min)
    if ohlcv is None:
        st.warning(f"No data for {symbol} {interval_min}min. Run the data bootstrap first.")
        return

    ohlcv = ohlcv.iloc[-bars:]

    features = None
    signals_df = None
    if show_signals:
        features = _load_features(symbol, interval_min)
        if features is not None:
            features = features.reindex(ohlcv.index)
            try:
                from production.strategy.momentum import MomentumStrategy, load_config
                close = ohlcv["close"]
                signals_df = MomentumStrategy(load_config()).generate_signals_df(
                    features.dropna(), close, vix=None
                )
            except Exception:
                pass

    ema_periods = [p for p, show in [(9, ema_9), (21, ema_21), (50, ema_50), (200, ema_200)] if show]

    from production.monitoring.components.charts import candlestick_chart
    fig = candlestick_chart(
        ohlcv=ohlcv,
        signals_df=signals_df,
        symbol=symbol,
        interval_min=interval_min,
        show_emas=ema_periods,
        show_volume=show_vol,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Stats strip ───────────────────────────────────────────────────────────
    st.divider()
    last  = ohlcv.iloc[-1]
    prev  = ohlcv.iloc[-2]
    chg   = last["close"] - prev["close"]
    chg_p = chg / prev["close"] * 100
    hi52  = ohlcv["high"].tail(252).max()
    lo52  = ohlcv["low"].tail(252).min()

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Last",      f"₹{last['close']:,.2f}", f"{chg:+.2f} ({chg_p:+.2f}%)")
    s2.metric("Open",      f"₹{last['open']:,.2f}")
    s3.metric("High",      f"₹{last['high']:,.2f}")
    s4.metric("Low",       f"₹{last['low']:,.2f}")
    s5.metric("52W High",  f"₹{hi52:,.2f}")
    s6.metric("52W Low",   f"₹{lo52:,.2f}")

    # ── Signal summary ────────────────────────────────────────────────────────
    if signals_df is not None and not signals_df.empty:
        st.divider()
        st.subheader("Signal Summary")
        sig_count = (signals_df["direction"] != 0).sum()
        long_count = (signals_df["direction"] == 1).sum()
        short_count = (signals_df["direction"] == -1).sum()

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total Signals", sig_count)
        sc2.metric("Long Signals",  long_count)
        sc3.metric("Short Signals", short_count)

        # Recent signals table
        recent_sigs = signals_df[signals_df["direction"] != 0].tail(10).copy()
        if not recent_sigs.empty:
            recent_sigs = recent_sigs.reset_index()
            # Index column name varies (could be 'datetime', 'index', etc.) — normalize to "Time"
            idx_col = recent_sigs.columns[0]
            recent_sigs = recent_sigs.rename(columns={idx_col: "Time"})
            recent_sigs["Time"]       = recent_sigs["Time"].astype(str).str[:16]
            recent_sigs["direction"]  = recent_sigs["direction"].map({1: "🟢 LONG", -1: "🔴 SHORT"})
            recent_sigs["confidence"] = recent_sigs["confidence"].apply(lambda x: f"{x:.0%}" if x > 0 else "—")
            recent_sigs["entry"]  = recent_sigs["entry"].apply(lambda x: f"₹{x:,.2f}")
            recent_sigs["stop"]   = recent_sigs["stop"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
            recent_sigs["target"] = recent_sigs["target"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
            show_cols = ["Time", "direction", "confidence", "entry", "stop", "target"]
            show_cols = [c for c in show_cols if c in recent_sigs.columns]
            display_df = recent_sigs[show_cols].rename(columns={
                "direction": "Direction", "confidence": "Confidence",
                "entry": "Entry Price", "stop": "Stop Loss", "target": "Target",
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── OHLCV data table ──────────────────────────────────────────────────────
    with st.expander("📋 Raw OHLCV Data (last 50 bars)"):
        display = ohlcv.tail(50).copy()
        for col in ["open", "high", "low", "close"]:
            display[col] = display[col].apply(lambda x: f"₹{x:,.2f}")
        display["volume"] = display["volume"].apply(lambda x: f"{x:,.0f}")
        display.index = display.index.strftime("%d %b %H:%M")
        st.dataframe(display[::-1], use_container_width=True)
