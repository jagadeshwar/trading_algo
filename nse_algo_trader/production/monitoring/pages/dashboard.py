"""Dashboard page — live P&L, positions, signals, system health."""

from __future__ import annotations
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

IST     = ZoneInfo("Asia/Kolkata")
PROJECT = Path(__file__).parents[3]   # nse_algo_trader/


def _next_expiry(symbol: str) -> str:
    """Next NSE weekly expiry (Thursday) for index symbols, monthly for equities."""
    today = date.today()
    days_ahead = (3 - today.weekday()) % 7   # 3 = Thursday
    if days_ahead == 0:
        days_ahead = 7
    expiry = today + timedelta(days=days_ahead)
    if "-EQ" in symbol:
        # Monthly: last Thursday of the month
        # Find last Thursday in the current month
        import calendar
        year, month = today.year, today.month
        last_day = calendar.monthrange(year, month)[1]
        for d in range(last_day, last_day - 7, -1):
            if date(year, month, d).weekday() == 3:
                expiry = date(year, month, d)
                if expiry <= today:   # already passed — go to next month
                    if month == 12:
                        year, month = year + 1, 1
                    else:
                        month += 1
                    last_day = calendar.monthrange(year, month)[1]
                    for dd in range(last_day, last_day - 7, -1):
                        if date(year, month, dd).weekday() == 3:
                            expiry = date(year, month, dd)
                            break
                break
    return expiry.strftime("%d %b")


@st.cache_data(ttl=15)
def _fetch_live_prices(symbols: tuple) -> dict:
    """Fetch live LTP for each symbol from Fyers. Cached 15s to avoid API spam."""
    try:
        import sys
        sys.path.insert(0, str(PROJECT))
        from dotenv import load_dotenv
        load_dotenv(PROJECT / ".env")
        from auth import get_fyers_client
        fyers = get_fyers_client()
        prices = {}
        for sym in symbols:
            resp = fyers.quotes({"symbols": sym})
            if resp.get("s") == "ok":
                prices[sym] = float(resp["d"][0]["v"]["lp"])
        prices["_at"] = datetime.now(IST).strftime("%H:%M:%S")
        return prices
    except Exception:
        return {}


def _live_equity(state: dict) -> tuple[float, float, str]:
    """Return (live_equity, unrealized_pnl, fetched_at) using live prices."""
    positions = state.get("positions", [])
    capital   = float(state.get("capital", 0))
    if not positions:
        # No open positions — equity = capital + closed P&L from JSONL
        trades_file = PROJECT / "data/trades_log.jsonl"
        closed_pnl = 0.0
        if trades_file.exists():
            try:
                rows = [json.loads(l) for l in trades_file.read_text().splitlines() if l.strip()]
                closed_pnl = sum(float(r.get("pnl", 0)) - float(r.get("costs", 0))
                                 for r in rows if r.get("type") == "EXIT")
            except Exception:
                pass
        return capital + closed_pnl, 0.0, "—"

    symbols = tuple(p["symbol"] for p in positions)
    live    = _fetch_live_prices(symbols)
    fetched_at = live.get("_at", "—")

    unrealized = 0.0
    for pos in positions:
        sym   = pos["symbol"]
        price = live.get(sym, pos.get("entry", 0))
        entry = float(pos.get("entry", 0))
        qty   = int(pos.get("qty", 0))
        dirn  = 1 if pos.get("direction") == "LONG" else -1
        unrealized += dirn * (price - entry) * qty

    # Equity = cash held + unrealized MTM + closed P&L
    trades_file = PROJECT / "data/trades_log.jsonl"
    closed_pnl = 0.0
    if trades_file.exists():
        try:
            rows = [json.loads(l) for l in trades_file.read_text().splitlines() if l.strip()]
            closed_pnl = sum(float(r.get("pnl", 0)) - float(r.get("costs", 0))
                             for r in rows if r.get("type") == "EXIT")
        except Exception:
            pass

    live_equity = capital + closed_pnl + unrealized
    return live_equity, unrealized, fetched_at


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
        trades_file = PROJECT / "data/trades_log.jsonl"
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
        sig_file = PROJECT / "data/signals_log.jsonl"
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
    notif_file = PROJECT / "data/.latest_notification.json"
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
            now_str = datetime.now(IST).strftime("%H:%M:%S")
            st.success(f"🟢 **LIVE** — {syms} · {tc_state.get('interval')}min · bar update {tc_state.get('last_update','—')} · prices as of {now_str}")
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

    trades  = _load_trades_from_db()
    signals = _load_signals_from_db()
    stats   = _compute_pnl_from_trades(trades)

    # ── Live equity from real-time prices ─────────────────────────────────────
    live_equity, unrealized_pnl, prices_at = _live_equity(tc_state)
    starting_capital = float(tc_state.get("capital", live_equity))
    live_total_pnl   = live_equity - starting_capital
    live_pnl_pct     = live_total_pnl / starting_capital * 100 if starting_capital else 0

    # ── P&L Hero Cards ────────────────────────────────────────────────────────
    st.subheader("Profit & Loss")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Live Equity",   f"₹{live_equity:,.0f}",
                  f"{live_pnl_pct:+.2f}% vs start", delta_color="normal")
    with c2:
        st.metric("Total P&L",     f"₹{live_total_pnl:+,.0f}",
                  f"Unrealized ₹{unrealized_pnl:+,.0f}" if unrealized_pnl else None,
                  delta_color="normal")
    with c3:
        st.metric("Total Trades",  stats["total_trades"])
    with c4:
        st.metric("Win Rate",      f"{stats['win_rate']}%",
                  "✓ above target" if stats["win_rate"] >= 52 else "✗ below 52%")
    with c5:
        st.metric("Profit Factor", f"{stats['profit_factor']}",
                  "✓ good" if stats["profit_factor"] >= 1.4 else "needs improvement")

    st.caption(f"Prices fetched at **{prices_at}** · Costs paid: ₹{stats['total_costs']:,.0f}")
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

    # ── Trade Journal ─────────────────────────────────────────────────────────
    tj_hdr, tj_btn = st.columns([4, 1])
    tj_hdr.subheader("Trade Journal")
    if tj_btn.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    trades_file = PROJECT / "data/trades_log.jsonl"
    if not trades_file.exists():
        st.info("No trades yet. Start paper trading — every entry and exit will appear here automatically.")
    else:
        try:
            raw_rows = [json.loads(l) for l in trades_file.read_text().splitlines() if l.strip()]
            if not raw_rows:
                st.info("No trades yet.")
            else:
                # Fetch live prices for all unique symbols in the journal
                unique_syms = tuple({r["symbol"] for r in raw_rows})
                live = _fetch_live_prices(unique_syms)
                fetched_at = live.get("_at", "—")

                journal = []
                for r in reversed(raw_rows):
                    trade_type = r.get("type", "")
                    pnl        = float(r.get("pnl", 0))
                    sym_raw    = r.get("symbol", "")
                    symbol     = sym_raw.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
                    direction  = r.get("direction", "")
                    cap_after  = float(r.get("capital_after", 0))
                    live_price = live.get(sym_raw)
                    expiry     = _next_expiry(sym_raw)

                    if trade_type == "ENTRY":
                        entry_px = float(r.get("price", 0))
                        qty      = int(r.get("qty", 0))
                        deployed = entry_px * qty
                        dirn     = 1 if direction == "LONG" else -1
                        live_mtm = (dirn * (live_price - entry_px) * qty) if live_price else None
                        journal.append({
                            "Time":          pd.to_datetime(r["time"]).strftime("%d %b %H:%M"),
                            "Type":          "🟢 LONG" if direction == "LONG" else "🔴 SHORT",
                            "Symbol":        symbol,
                            "Qty":           qty,
                            "Entry Price":   f"₹{entry_px:,.2f}",
                            "Current Price": f"₹{live_price:,.2f}" if live_price else "—",
                            "Stop":          f"₹{float(r.get('stop',0)):,.2f}",
                            "Target":        f"₹{float(r.get('target',0)):,.2f}",
                            "Expiry":        expiry,
                            "Deployed":      f"₹{deployed:,.0f}",
                            "Unrealized":    f"₹{live_mtm:+,.0f}" if live_mtm is not None else "—",
                            "P&L":           "—",
                            "Capital":       f"₹{cap_after:,.0f}" if cap_after else "—",
                            "Status":        "🔵 Open",
                        })
                    elif trade_type == "EXIT":
                        result = "✅ Profit" if pnl > 0 else ("❌ Loss" if pnl < 0 else "➖ B/E")
                        reason = r.get("reason", "").split()[0] if r.get("reason") else "—"
                        journal.append({
                            "Time":          pd.to_datetime(r["time"]).strftime("%d %b %H:%M"),
                            "Type":          "🏁 EXIT",
                            "Symbol":        symbol,
                            "Qty":           r.get("qty"),
                            "Entry Price":   f"₹{float(r.get('entry_price',0)):,.2f}" if r.get('entry_price') else "—",
                            "Current Price": f"₹{live_price:,.2f}" if live_price else "—",
                            "Stop":          "—",
                            "Target":        "—",
                            "Expiry":        expiry,
                            "Deployed":      "—",
                            "Unrealized":    "—",
                            "P&L":           f"₹{pnl:+,.0f}",
                            "Capital":       f"₹{cap_after:,.0f}" if cap_after else "—",
                            "Status":        f"{result} · {reason}",
                        })

                df_journal = pd.DataFrame(journal)

                def _style_cell(val):
                    if not isinstance(val, str) or val in ("—", ""):
                        return ""
                    try:
                        v = float(val.replace("₹","").replace(",","").replace("+",""))
                        return "color: #00c896; font-weight:600" if v > 0 else \
                               ("color: #ff4c6a; font-weight:600" if v < 0 else "")
                    except Exception:
                        return ""

                styled = df_journal.style\
                    .applymap(_style_cell, subset=["P&L", "Unrealized"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
                st.caption(f"Current prices fetched at **{fetched_at}** IST · click 🔄 Refresh for latest")

                # Summary row
                all_exits  = [r for r in raw_rows if r.get("type") == "EXIT"]
                total_pnl  = sum(float(r.get("pnl", 0)) for r in all_exits)
                total_cost = sum(float(r.get("costs", 0)) for r in raw_rows)
                wins   = [r for r in all_exits if float(r.get("pnl", 0)) > 0]
                losses = [r for r in all_exits if float(r.get("pnl", 0)) < 0]

                st.divider()
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Closed Trades", len(all_exits))
                sc2.metric("Net P&L",       f"₹{total_pnl:+,.0f}",
                           "profit" if total_pnl > 0 else ("loss" if total_pnl < 0 else None))
                sc3.metric("Win / Loss",    f"{len(wins)}W / {len(losses)}L")
                sc4.metric("Total Costs",   f"₹{total_cost:,.0f}")
        except Exception as e:
            st.warning(f"Could not load trade journal: {e}")

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
