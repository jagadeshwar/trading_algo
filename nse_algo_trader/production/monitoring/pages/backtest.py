"""Backtest page — run strategy on historical data and view performance report."""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

SYMBOLS = [
    "NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX",
    "NSE:HDFCBANK-EQ", "NSE:RELIANCE-EQ", "NSE:ICICIBANK-EQ",
    "NSE:TCS-EQ", "NSE:SBIN-EQ", "NSE:AXISBANK-EQ",
    "NSE:KOTAKBANK-EQ", "NSE:LT-EQ", "NSE:ITC-EQ",
]
INTERVALS = {5: "5 min", 15: "15 min", 1440: "Daily"}
TARGET_METRICS = {"Sharpe Ratio": 1.5, "Win Rate %": 52.0, "Profit Factor": 1.4, "Max DD %": 10.0}


def render():
    st.title("🧪 Backtest")
    st.caption("Run the momentum strategy on historical data with full NSE transaction costs.")

    # ── Controls ──────────────────────────────────────────────────────────────
    with st.form("backtest_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            symbol = st.selectbox("Symbol", SYMBOLS, index=3)
        with c2:
            interval = st.selectbox("Timeframe", list(INTERVALS.keys()),
                                    format_func=lambda x: INTERVALS[x], index=1)
        with c3:
            capital = st.number_input("Capital (₹)", min_value=100_000,
                                       max_value=50_000_000, value=1_000_000, step=100_000)

        st.markdown("**Risk targets (for live-ready check):**")
        tc1, tc2, tc3, tc4 = st.columns(4)
        sharpe_target = tc1.number_input("Min Sharpe", 0.5, 5.0, 1.5, 0.1)
        dd_target     = tc2.number_input("Max DD %",   1.0, 30.0, 10.0, 1.0)
        wr_target     = tc3.number_input("Min Win %",  40.0, 70.0, 52.0, 1.0)
        pf_target     = tc4.number_input("Min PF",     0.5, 5.0, 1.4, 0.1)

        run_btn = st.form_submit_button("▶️  Run Backtest", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Configure parameters above and click **Run Backtest** to start.")

        # Show previously cached result if available
        if "bt_result" in st.session_state:
            st.divider()
            st.caption("Showing last backtest result:")
            _render_results(st.session_state.bt_result, st.session_state.bt_signals,
                            st.session_state.bt_close, capital)
        return

    # ── Run backtest ──────────────────────────────────────────────────────────
    with st.spinner(f"Running backtest on {symbol} {interval}min..."):
        try:
            from production.strategy.backtest import (
                load_ohlcv_and_features, _manual_backtest
            )
            from production.strategy.momentum import MomentumStrategy, load_config
            import pyarrow.parquet as pq, yaml
            from pathlib import Path
            import numpy as np

            ohlcv, feats = load_ohlcv_and_features(symbol, interval)
            common = feats.index.intersection(ohlcv.index)
            feats = feats.loc[common]
            close = ohlcv.loc[common, "close"]

            vix_series = None
            try:
                vpath = Path(f"data/ohlcv/NSE_INDIAVIX_INDEX_{interval}min.parquet")
                if vpath.exists():
                    vdf = pq.read_table(vpath).to_pandas()
                    vdf.index = pd.to_datetime(vdf.index, utc=True).tz_convert("Asia/Kolkata")
                    vix_series = vdf["close"].reindex(common)
            except Exception:
                pass

            signals_df = MomentumStrategy(load_config()).generate_signals_df(feats, close, vix=vix_series)
            costs_cfg = yaml.safe_load(Path("configs/broker.yaml").read_text()).get("transaction_costs", {})
            fee = costs_cfg.get("exchange_fee_pct", 0.00345) / 100 + costs_cfg.get("slippage_pct", 0.02) / 100

            result = _manual_backtest(signals_df, close, capital, fee, interval, symbol=symbol)
            result.update({"symbol": symbol, "interval_min": interval})

            st.session_state.bt_result  = result
            st.session_state.bt_signals = signals_df
            st.session_state.bt_close   = close

        except FileNotFoundError as e:
            st.error(f"Data not found: {e}. Run the data bootstrap first.")
            return
        except Exception as e:
            st.error(f"Backtest error: {e}")
            return

    _render_results(st.session_state.bt_result, st.session_state.bt_signals,
                    st.session_state.bt_close, capital,
                    targets={"sharpe": sharpe_target, "max_dd_pct": dd_target,
                             "win_rate_pct": wr_target, "profit_factor": pf_target})


def _render_results(result: dict, signals_df, close, capital: float, targets: dict | None = None):
    if targets is None:
        targets = {"sharpe": 1.5, "max_dd_pct": 10.0, "win_rate_pct": 52.0, "profit_factor": 1.4}

    st.divider()
    st.subheader(f"Results — {result.get('symbol', '')} {result.get('interval_min', '')}min")

    # ── Metric cards ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    sharpe = result.get("sharpe", 0)
    max_dd = result.get("max_dd_pct", 0)
    wr     = result.get("win_rate_pct", 0)
    pf     = result.get("profit_factor", 0)
    ret    = result.get("total_ret_pct", 0)

    m1.metric("Sharpe Ratio",  f"{sharpe:.3f}",
              "✓ Pass" if sharpe >= targets["sharpe"] else "✗ Below target",
              delta_color="normal" if sharpe >= targets["sharpe"] else "inverse")
    m2.metric("Max Drawdown",  f"{max_dd:.2f}%",
              "✓ Pass" if max_dd <= targets["max_dd_pct"] else "✗ Too high",
              delta_color="normal" if max_dd <= targets["max_dd_pct"] else "inverse")
    m3.metric("Win Rate",      f"{wr:.1f}%",
              "✓ Pass" if wr >= targets["win_rate_pct"] else "✗ Below 52%",
              delta_color="normal" if wr >= targets["win_rate_pct"] else "inverse")
    m4.metric("Profit Factor", f"{pf:.3f}",
              "✓ Pass" if pf >= targets["profit_factor"] else "✗ Below 1.4",
              delta_color="normal" if pf >= targets["profit_factor"] else "inverse")
    m5.metric("Total Return",  f"{ret:.2f}%",
              f"{result.get('total_trades', 0)} trades")

    live_ready = result.get("live_ready", False)
    if live_ready:
        st.success("🚀  **LIVE READY** — All targets met. Strategy cleared for paper trading.")
    else:
        passed = sum([
            sharpe >= targets["sharpe"],
            max_dd <= targets["max_dd_pct"],
            wr >= targets["win_rate_pct"],
            pf >= targets["profit_factor"],
        ])
        st.warning(f"⚠️  **NOT READY** — {passed}/4 targets met. Continue paper trading and tuning.")

    st.divider()

    # ── Targets reference ─────────────────────────────────────────────────────
    with st.expander("📋 Target thresholds explained"):
        st.markdown("""
| Metric | Target | Meaning |
|--------|--------|---------|
| **Sharpe Ratio** | > 1.5 | Risk-adjusted return. > 1.5 = good, > 2.0 = excellent |
| **Max Drawdown** | < 10% | Worst peak-to-trough loss. Keep below 10% before going live |
| **Win Rate** | > 52% | % of trades that are profitable |
| **Profit Factor** | > 1.4 | Gross profit ÷ Gross loss. 1.4 = win ₹1.40 for every ₹1 lost |
        """)

    # ── Equity curve ──────────────────────────────────────────────────────────
    if signals_df is not None and close is not None:
        st.subheader("Equity Curve")
        _plot_equity(signals_df, close, capital)

    # ── Signal count ──────────────────────────────────────────────────────────
    if signals_df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Bars analysed",  f"{result.get('bars', 0):,}")
        c2.metric("Signals fired",  f"{result.get('signals', 0)}")
        c3.metric("Trades executed",f"{result.get('total_trades', 0)}")


def _plot_equity(signals_df, close, capital: float) -> None:
    import numpy as np

    # Simulate equity from signals
    equity = [capital]
    times  = [signals_df.index[0]]
    position = None

    for ts, row in signals_df.iterrows():
        price = close.loc[ts]
        if position:
            d = position["dir"]
            if (d == 1 and price <= position["stop"]) or (d == -1 and price >= position["stop"]):
                pnl = d * (price - position["entry"]) * position["qty"]
                equity.append(equity[-1] + pnl)
                times.append(ts)
                position = None
            elif (d == 1 and price >= position["target"]) or (d == -1 and price <= position["target"]):
                pnl = d * (price - position["entry"]) * position["qty"]
                equity.append(equity[-1] + pnl)
                times.append(ts)
                position = None

        if position is None and row.get("direction", 0) != 0:
            qty = max(1, int(capital * 0.05 / price))
            position = {"dir": int(row["direction"]), "entry": price,
                        "stop": float(row["stop"]) if pd.notna(row.get("stop")) else price * 0.98,
                        "target": float(row["target"]) if pd.notna(row.get("target")) else price * 1.03,
                        "qty": qty}

    eq_series = pd.Series(equity, index=times)
    ret_pct = (eq_series / capital - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ret_pct.index, y=ret_pct.values,
        fill="tozeroy",
        fillcolor=f"rgba({'0,200,150' if ret_pct.iloc[-1] >= 0 else '255,76,106'},0.15)",
        line=dict(color="#00c896" if ret_pct.iloc[-1] >= 0 else "#ff4c6a", width=2),
        name="Return %",
    ))
    fig.add_hline(y=0, line_color="#444", line_dash="dash")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        yaxis_title="Return (%)", height=300,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
