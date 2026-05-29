"""Dashboard page — live P&L, positions, signals, system health."""

from __future__ import annotations
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

IST = ZoneInfo("Asia/Kolkata")


def _load_trades_from_db() -> pd.DataFrame:
    try:
        from production.db.database import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT o.created_at, o.symbol, o.side, o.qty, o.price,
                       o.status, o.strategy_version, o.reject_reason,
                       f.fill_price, f.slippage_pct,
                       f.total_cost,
                       COALESCE(f.qty_filled, 0) AS qty_filled
                FROM orders o
                LEFT JOIN fills f ON f.order_id = o.id
                WHERE o.broker = 'paper'
                ORDER BY o.created_at DESC
                LIMIT 100
            """), conn)
        if not df.empty:
            return df
    except Exception:
        pass
    # Fallback: read from JSONL file written by run_paper_trading.py
    try:
        trades_file = Path("data/trades_log.jsonl")
        if not trades_file.exists():
            return pd.DataFrame()
        rows = [json.loads(l) for l in trades_file.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(rows[::-1][:100])
        # Normalise columns to match DB schema expected by _compute_pnl_from_trades
        df = df.rename(columns={"time": "created_at", "price": "fill_price"})
        df["status"] = "FILLED"
        df["qty_filled"] = df["qty"]
        df["total_cost"] = df.get("costs", 0)
        return df
    except Exception:
        return pd.DataFrame()


def _load_signals_from_db(limit: int = 20) -> pd.DataFrame:
    try:
        from production.db.database import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT time, symbol, direction, confidence, regime,
                       model_version, strategy_version, acted_on
                FROM signals
                ORDER BY time DESC
                LIMIT {limit}
            """), conn)
        if not df.empty:
            return df
    except Exception:
        pass
    # Fallback: read from JSONL file written by run_paper_trading.py
    try:
        sig_file = Path("data/signals_log.jsonl")
        if not sig_file.exists():
            return pd.DataFrame()
        rows = [json.loads(l) for l in sig_file.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(rows[::-1][:limit])
        return df
    except Exception:
        return pd.DataFrame()


def _compute_pnl_from_trades(trades: pd.DataFrame) -> dict:
    """Compute P&L stats from the orders+fills table."""
    if trades.empty or "fill_price" not in trades.columns:
        return {"total_pnl": 0, "total_pnl_pct": 0, "daily_pnl": 0,
                "total_trades": 0, "win_rate": 0, "profit_factor": 0, "total_costs": 0}

    filled = trades[trades["status"] == "FILLED"].copy()
    if filled.empty:
        return {"total_pnl": 0, "total_pnl_pct": 0, "daily_pnl": 0,
                "total_trades": 0, "win_rate": 0, "profit_factor": 0, "total_costs": 0}

    # Pair BUY and SELL for same symbol
    pnls = []
    for symbol, grp in filled.groupby("symbol"):
        buys  = grp[grp["side"] == "BUY"].sort_values("created_at")
        sells = grp[grp["side"] == "SELL"].sort_values("created_at")
        for (_, b), (_, s) in zip(buys.iterrows(), sells.iterrows()):
            qty = min(b["qty_filled"], s["qty_filled"])
            if qty > 0:
                pnl = (s["fill_price"] - b["fill_price"]) * qty
                pnls.append(pnl)

    if not pnls:
        return {"total_pnl": 0, "total_pnl_pct": 0, "daily_pnl": 0,
                "total_trades": 0, "win_rate": 0, "profit_factor": 0, "total_costs": 0}

    total_pnl = sum(pnls)
    wins  = [p for p in pnls if p > 0]
    losses= [p for p in pnls if p < 0]
    total_costs = filled["total_cost"].fillna(0).sum()
    today = date.today().isoformat()
    today_trades = filled[filled["created_at"].astype(str).str[:10] == today]
    daily_pnl = today_trades["total_cost"].fillna(0).sum() * -1  # rough proxy

    return {
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl / 1_000_000 * 100, 3),
        "daily_pnl": round(daily_pnl, 2),
        "total_trades": len(pnls),
        "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else 0,
        "total_costs": round(total_costs, 2),
    }


def _show_toast() -> None:
    """Show Streamlit toast for latest unread notification."""
    notif_file = Path("data/.latest_notification.json")
    if not notif_file.exists():
        return
    try:
        n = json.loads(notif_file.read_text())
        if n.get("read"):
            return
        event = n.get("event", "")
        icon_map = {"EXIT_WIN": "✅", "EXIT_LOSS": "❌", "ENTRY": "🟢",
                    "CIRCUIT": "🚨", "DAILY": "📊", "TEST": "🔔"}
        emoji = icon_map.get(event, "🔔")
        # Show toast
        st.toast(f"{emoji} **{n.get('title','')}**\n\n{n.get('body','')}", icon=emoji)
        # Mark as read
        n["read"] = True
        notif_file.write_text(json.dumps(n))
    except Exception:
        pass


def render():
    import os, signal, time
    from production.monitoring.pages.trading_controls import (
        _is_running, _start, _stop, _read_state as _tc_state
    )

    st.title("📊 Live Dashboard")
    _show_toast()

    # ── Trading status + quick-toggle ─────────────────────────────────────────
    running = _is_running()
    tc_state = _tc_state()

    status_col, btn_col, refresh_col, interval_col = st.columns([3, 1, 1, 1])
    with status_col:
        if running:
            syms = " · ".join(s.replace("NSE:","").replace("-EQ","").replace("-INDEX","")
                              for s in tc_state.get("symbols", []))
            st.success(f"🟢 **LIVE** — {syms} · {tc_state.get('interval')}min · updated {tc_state.get('last_update','—')}")
        else:
            st.warning("🔴 **Paper trading is stopped**")
    with btn_col:
        if running:
            if st.button("⏹️ Stop", type="primary", use_container_width=True):
                _stop(); time.sleep(1); st.rerun()
        else:
            if st.button("▶️ Start", type="primary", use_container_width=True):
                prev = tc_state
                syms = prev.get("symbols") or ["NSE:HDFCBANK-EQ"]
                _start(syms, prev.get("interval", 15), float(prev.get("capital", 1_000_000)))
                time.sleep(2); st.rerun()
    with refresh_col:
        auto = st.toggle("Auto-refresh", value=False)
    with interval_col:
        interval = st.selectbox("Every", [5, 10, 30, 60], index=1,
                                format_func=lambda x: f"{x}s", label_visibility="collapsed")
    if auto:
        time.sleep(interval)
        st.rerun()
    st.divider()

    trades = _load_trades_from_db()
    signals = _load_signals_from_db()
    stats = _compute_pnl_from_trades(trades)

    # ── P&L Hero Cards ────────────────────────────────────────────────────────
    st.subheader("Profit & Loss")
    c1, c2, c3, c4, c5 = st.columns(5)

    total_pnl = stats["total_pnl"]
    daily_pnl = stats["daily_pnl"]

    with c1:
        st.metric(
            "Total P&L",
            f"₹{total_pnl:+,.0f}",
            f"{stats['total_pnl_pct']:+.3f}%",
            delta_color="normal",
        )
    with c2:
        st.metric("Today's P&L", f"₹{daily_pnl:+,.0f}")
    with c3:
        st.metric("Total Trades", stats["total_trades"])
    with c4:
        st.metric("Win Rate", f"{stats['win_rate']}%",
                  "✓ above target" if stats["win_rate"] >= 52 else "✗ below 52%")
    with c5:
        st.metric("Profit Factor", f"{stats['profit_factor']}",
                  "✓ good" if stats["profit_factor"] >= 1.4 else "needs improvement")

    st.caption(f"Transaction costs paid: ₹{stats['total_costs']:,.0f}")
    st.divider()

    # ── P&L Explanation ───────────────────────────────────────────────────────
    with st.expander("ℹ️  What do these numbers mean?", expanded=False):
        st.markdown("""
| Term | Meaning |
|------|---------|
| **Total P&L** | Net profit or loss across ALL paper trades since start (in ₹) |
| **Today's P&L** | Profit or loss from trades opened/closed today only |
| **Win Rate** | % of trades that closed in profit. Target > 52% |
| **Profit Factor** | Total wins ÷ Total losses. Target > 1.4 (e.g. 1.4 = ₹1.40 won for every ₹1 lost) |
| **Transaction Costs** | Brokerage + STT + exchange fees + slippage paid. Deducted from P&L |

**P&L = (Sum of all winning trades) − (Sum of all losing trades) − Transaction costs**
        """)

    # ── Recent signals ────────────────────────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Recent Signals")
        if signals.empty:
            if running:
                st.info("⏳ Paper trading is ACTIVE — waiting for the strategy to fire a signal.\n\n"
                        "Signals fire on either:\n"
                        "- EMA 9/21 crossover with ADX > 25 + volume > 1.2×\n"
                        "- Strong trend continuation (ADX > 40 + DI strongly directional)\n\n"
                        "Signals will appear here automatically when fired.")
            else:
                st.info("No signals yet. Start paper trading from 🎮 Trading Controls.")
        else:
            display = signals.copy()
            display["direction"] = display["direction"].apply(
                lambda x: "🟢 LONG" if int(x) == 1 else ("🔴 SHORT" if int(x) == -1 else "⬜ FLAT")
            )
            display["confidence"] = display["confidence"].apply(lambda x: f"{float(x):.0%}")
            display["acted_on"]   = display.get("acted_on", pd.Series([True]*len(display))).apply(
                lambda x: "✅ Filled" if x else "⏭ Skipped"
            )
            display["time"] = pd.to_datetime(display["time"]).dt.strftime("%d %b %H:%M")
            display = display.rename(columns={
                "time": "Time", "symbol": "Symbol", "direction": "Direction",
                "confidence": "Confidence", "regime": "Regime", "acted_on": "Result",
            })
            cols = [c for c in ["Time", "Symbol", "Direction", "Confidence", "Regime", "Result"] if c in display.columns]
            st.dataframe(display[cols], use_container_width=True, hide_index=True)

    with right:
        st.subheader("Recent Trades")
        if trades.empty:
            st.info("No paper trades logged yet.")
        else:
            display = trades.head(20).copy()
            display["created_at"] = pd.to_datetime(display["created_at"]).dt.strftime("%d %b %H:%M")
            display["side"] = display["side"].map({"BUY": "🟢 BUY", "SELL": "🔴 SELL"})
            display["fill_price"] = display["fill_price"].apply(
                lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
            display = display.rename(columns={
                "created_at": "Time", "symbol": "Symbol",
                "side": "Side", "qty": "Qty", "fill_price": "Fill Price", "status": "Status",
            })
            st.dataframe(
                display[["Time", "Symbol", "Side", "Qty", "Fill Price", "Status"]],
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── Per-trade P&L breakdown ────────────────────────────────────────────────
    st.subheader("Per-Trade P&L Breakdown")
    st.caption("Each row = one completed trade (entry + exit pair). Green = profit, Red = loss.")

    # Build per-trade table from log files (most reliable source)
    log_dir = Path("logs")
    paper_logs = sorted(log_dir.glob("paper_trading_*.log"), reverse=True)[:3]
    trade_rows = []
    for log_file in paper_logs:
        for line in log_file.read_text().splitlines():
            if "PAPER EXIT" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    trade_rows.append({"raw": parts[-1].strip()})

    if trade_rows:
        # Parse "PAPER EXIT NSE:HDFCBANK-EQ 10:45 @ 998.0  PnL=+582  reason=TARGET  ✓"
        parsed = []
        for t in trade_rows:
            r = t["raw"]
            try:
                parts = r.replace("PAPER EXIT ", "").split()
                symbol = parts[0]
                time_  = parts[1]
                price  = parts[3]
                pnl_str = [p for p in parts if p.startswith("PnL=")][0].replace("PnL=", "")
                reason = [p for p in parts if p.startswith("reason=")][0].replace("reason=", "")
                pnl_val = float(pnl_str.replace(",", "").replace("₹", ""))
                parsed.append({"Symbol": symbol, "Time": time_, "Exit Price": price,
                               "P&L (₹)": pnl_val, "Reason": reason,
                               "Result": "✅ Profit" if pnl_val > 0 else "❌ Loss"})
            except Exception:
                pass

        if parsed:
            df_trades = pd.DataFrame(parsed)
            df_trades["P&L (₹)"] = df_trades["P&L (₹)"].apply(lambda x: f"₹{x:+,.0f}")
            st.dataframe(df_trades, use_container_width=True, hide_index=True)
        else:
            st.info("Trade details will appear here once paper trading has run.")
    else:
        st.info("No trade logs found yet. Start paper trading to see detailed P&L here.")

    # ── Data stats ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Data Store")
    try:
        from production.db.database import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            ohlcv_count = conn.execute(text("SELECT COUNT(*) FROM ohlcv")).scalar()
            feat_count  = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()
            sig_count   = conn.execute(text("SELECT COUNT(*) FROM signals")).scalar()
            ord_count   = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar()

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("OHLCV Rows",    f"{ohlcv_count:,}")
        d2.metric("Feature Rows",  f"{feat_count:,}")
        d3.metric("Signals Logged", f"{sig_count:,}")
        d4.metric("Orders Logged",  f"{ord_count:,}")
    except Exception:
        st.warning("Database not connected — run PostgreSQL to see data stats.")
