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


def _fetch_live_prices(symbols: tuple) -> dict:
    """Fetch live LTP via session-state cache (works correctly with st.rerun)."""
    import time as _t
    cache_key = f"lp_cache_{hash(symbols)}"
    time_key  = f"lp_time_{hash(symbols)}"
    now = _t.time()

    cached    = st.session_state.get(cache_key, {})
    last_time = st.session_state.get(time_key, 0)

    if cached and (now - last_time) < 4:   # reuse if < 4s old
        return cached

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
    except Exception:
        prices = cached   # keep stale on error

    st.session_state[cache_key] = prices
    st.session_state[time_key]  = now
    return prices


def _fetch_atm_options(symbol: str, ref_price: float) -> dict:
    """Fetch ATM CE and PE for the nearest weekly expiry. Session-state cached 30s."""
    import time as _t
    step      = 100 if "NIFTYBANK" in symbol else 50
    atm       = round(ref_price / step) * step
    cache_key = f"opts_{symbol}_{atm}"
    time_key  = f"opts_t_{symbol}_{atm}"
    now       = _t.time()

    cached    = st.session_state.get(cache_key, {})
    if cached and (now - st.session_state.get(time_key, 0)) < 30:
        return cached

    result = {"atm_strike": atm, "ce_ltp": None, "pe_ltp": None,
              "ce_sym": "—", "pe_sym": "—", "expiry": "—"}
    try:
        import sys; sys.path.insert(0, str(PROJECT))
        from dotenv import load_dotenv; load_dotenv(PROJECT / ".env")
        from auth import get_fyers_client
        from production.data.fetcher import DataFetcher
        chain = DataFetcher(get_fyers_client()).fetch_options_chain(symbol, strike_count=5)
        if not chain.empty:
            result["expiry"] = chain["expiry"].iloc[0]
            atm_rows = chain[chain["strike"] == int(atm)]   # ensure int match
            ce = atm_rows[atm_rows["option_type"] == "CE"]
            pe = atm_rows[atm_rows["option_type"] == "PE"]
            if not ce.empty:
                result["ce_ltp"] = float(ce["ltp"].iloc[0])
                result["ce_sym"] = ce["symbol"].iloc[0]
            if not pe.empty:
                result["pe_ltp"] = float(pe["ltp"].iloc[0])
                result["pe_sym"] = pe["symbol"].iloc[0]
    except Exception:
        result = cached or result

    # Only cache if we got real data — never cache empty results
    if result.get("ce_ltp") or result.get("pe_ltp"):
        st.session_state[cache_key] = result
        st.session_state[time_key]  = now
    return result


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
            if st.button("⏹️ Stop", type="primary", width="stretch"):
                _stop(); time.sleep(1); st.rerun()
        else:
            if st.button("▶️ Start", type="primary", width="stretch"):
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

    # Closed P&L from JSONL (realised only)
    trades_file = PROJECT / "data/trades_log.jsonl"
    closed_pnl  = 0.0
    closed_costs = 0.0
    closed_count = 0
    try:
        if trades_file.exists():
            rows = [json.loads(l) for l in trades_file.read_text().splitlines() if l.strip()]
            for r in rows:
                if r.get("type") == "EXIT":
                    closed_pnl   += float(r.get("pnl", 0))
                    closed_costs += float(r.get("costs", 0))
                    closed_count += 1
                elif r.get("type") == "ENTRY":
                    closed_costs += float(r.get("costs", 0))
    except Exception:
        pass

    # ── P&L Hero Cards ────────────────────────────────────────────────────────
    st.subheader("Profit & Loss")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Live Equity",    f"₹{live_equity:,.0f}",
                  f"{live_pnl_pct:+.2f}% vs start", delta_color="normal")
    with c2:
        st.metric("Realised P&L",   f"₹{closed_pnl:+,.0f}",
                  f"net of costs ₹{(closed_pnl - closed_costs):+,.0f}",
                  delta_color="normal")
    with c3:
        st.metric("Unrealized P&L", f"₹{unrealized_pnl:+,.0f}" if unrealized_pnl else "₹0",
                  f"live @ {prices_at}" if prices_at != "—" else "no open positions",
                  delta_color="normal")
    with c4:
        st.metric("Win Rate",       f"{stats['win_rate']}%",
                  "✓ above target" if stats["win_rate"] >= 52 else "✗ below 52%")
    with c5:
        st.metric("Profit Factor",  f"{stats['profit_factor']}",
                  "✓ good" if stats["profit_factor"] >= 1.4 else "needs improvement")

    st.caption(f"Live prices as of **{prices_at}** IST · auto-refresh clears cache each cycle · costs paid ₹{closed_costs:,.0f}")
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
            display["time"] = pd.to_datetime(display["time"], format="ISO8601").dt.strftime("%d %b %H:%M")
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
            display = trades.copy()
            # Closed symbols set
            exited = set(display.loc[display["type"] == "EXIT", "symbol"]) \
                     if "type" in display.columns else set()

            # Fetch live prices and options for symbols in recent trades
            rt_syms  = tuple(display["symbol"].unique()) if "symbol" in display.columns else ()
            rt_live  = _fetch_live_prices(rt_syms) if rt_syms else {}
            rt_opts: dict[str, dict] = {}
            for sym in rt_syms:
                sp = rt_live.get(sym)
                if sp:
                    rt_opts[sym] = _fetch_atm_options(sym, sp)

            def _rt_status(row):
                t   = str(row.get("type", "ENTRY"))
                sym = row.get("symbol", "")
                if t == "EXIT":
                    pnl = float(row.get("pnl", 0))
                    return "✅ Closed · Profit" if pnl > 0 else \
                           ("❌ Closed · Loss" if pnl < 0 else "✅ Closed · B/E")
                return "✅ Closed" if sym in exited else "🟢 Open"

            def _rt_atm(row):
                sym = row.get("symbol", "")
                ref = float(row.get("fill_price", 0) or 0)
                step = 100 if "NIFTYBANK" in sym else 50
                return f"₹{round(ref / step) * step:,.0f}" if ref else "—"

            def _rt_ce(row):
                o = rt_opts.get(row.get("symbol",""), {})
                v = o.get("ce_ltp")
                return f"₹{v:,.2f}" if v else "—"

            def _rt_pe(row):
                o = rt_opts.get(row.get("symbol",""), {})
                v = o.get("pe_ltp")
                return f"₹{v:,.2f}" if v else "—"

            # Compute derived columns BEFORE formatting fill_price to string
            display["Status"]      = display.apply(_rt_status, axis=1)
            display["ATM Strike"]  = display.apply(_rt_atm, axis=1)
            display["Call(CE) ₹"]  = display.apply(_rt_ce, axis=1)
            display["Put(PE) ₹"]   = display.apply(_rt_pe, axis=1)
            # Format display columns last
            display["created_at"]  = pd.to_datetime(display["created_at"], format="ISO8601").dt.strftime("%d %b %H:%M")
            display["side"]        = display["side"].map({"BUY": "🟢 BUY", "SELL": "🔴 SELL"})
            display["fill_price"]  = display["fill_price"].apply(
                                         lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
            display = display.rename(columns={
                "created_at": "Time", "symbol": "Symbol",
                "side": "Side", "qty": "Qty", "fill_price": "Buy/Sell Price",
            })
            st.dataframe(
                display[["Time","Symbol","Side","Qty","Buy/Sell Price",
                          "ATM Strike","Call(CE) ₹","Put(PE) ₹","Status"]].head(20),
                use_container_width=True, hide_index=True,
            )
            st.caption("ATM Strike = exercise price · Call(CE)/Put(PE) = option premiums at that strike")

    st.divider()

    # ── Trade Journal ─────────────────────────────────────────────────────────
    tj_hdr, tj_btn = st.columns([4, 1])
    tj_hdr.subheader("Trade Journal")
    if tj_btn.button("🔄 Refresh", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("lp_") or k.startswith("opts_"):
                del st.session_state[k]
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
                # Fetch live prices for all unique symbols
                unique_syms = tuple({r["symbol"] for r in raw_rows})
                live        = _fetch_live_prices(unique_syms)
                fetched_at  = live.get("_at", "—")

                # Pre-fetch ATM options for each symbol (based on live spot)
                opts_cache: dict[str, dict] = {}
                for sym in unique_syms:
                    sp = live.get(sym)
                    if sp:
                        opts_cache[sym] = _fetch_atm_options(sym, sp)

                # Build set of closed symbols (symbol has an EXIT row)
                closed_symbols: set[str] = {
                    r["symbol"] for r in raw_rows if r.get("type") == "EXIT"
                }

                journal = []
                for r in reversed(raw_rows):
                    trade_type = r.get("type", "")
                    pnl        = float(r.get("pnl", 0))
                    sym_raw    = r.get("symbol", "")
                    symbol     = sym_raw.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
                    direction  = r.get("direction", "")
                    cap_after  = float(r.get("capital_after", 0))
                    live_price = live.get(sym_raw)

                    # Options always based on LIVE spot price (consistent, actionable)
                    opts       = opts_cache.get(sym_raw, {})
                    step       = 100 if "NIFTYBANK" in sym_raw else 50
                    live_atm   = round(live_price / step) * step if live_price else None
                    expiry     = opts.get("expiry", "—")
                    ce_ltp     = opts.get("ce_ltp")
                    pe_ltp     = opts.get("pe_ltp")

                    # Suggested options action based on direction
                    if trade_type == "ENTRY":
                        opt_action = ("📈 BUY CE @ ₹" + f"{ce_ltp:,.2f}" if direction == "LONG" and ce_ltp
                                      else "📉 BUY PE @ ₹" + f"{pe_ltp:,.2f}" if direction == "SHORT" and pe_ltp
                                      else "—")
                    else:
                        opt_action = ("📉 SELL PE @ ₹" + f"{pe_ltp:,.2f}" if direction == "SHORT" and pe_ltp
                                      else "📈 SELL CE @ ₹" + f"{ce_ltp:,.2f}" if direction == "LONG" and ce_ltp
                                      else "—")

                    if trade_type == "ENTRY":
                        entry_px  = float(r.get("price", 0))
                        qty       = int(r.get("qty", 0))
                        dirn      = 1 if direction == "LONG" else -1
                        live_mtm  = (dirn * (live_price - entry_px) * qty) if live_price else None
                        is_closed = sym_raw in closed_symbols
                        journal.append({
                            "Time":           pd.to_datetime(r["time"], format="ISO8601").strftime("%d %b %H:%M"),
                            "Type":           "🟢 LONG" if direction == "LONG" else "🔴 SHORT",
                            "Symbol":         symbol,
                            "Qty":            qty,
                            "Entry Price":    f"₹{entry_px:,.2f}",
                            "Live Price":     f"₹{live_price:,.2f}" if live_price else "—",
                            "ATM Strike":     f"₹{live_atm:,.0f}" if live_atm else "—",
                            "Call Price(CE)": f"₹{ce_ltp:,.2f}" if ce_ltp else "—",
                            "Put Price(PE)":  f"₹{pe_ltp:,.2f}" if pe_ltp else "—",
                            "Options Action": opt_action,
                            "Expiry":         expiry,
                            "Stop":           f"₹{float(r.get('stop',0)):,.2f}",
                            "Target":         f"₹{float(r.get('target',0)):,.2f}",
                            "Unrealized":     f"₹{live_mtm:+,.0f}" if live_mtm is not None else "—",
                            "Realised P&L":   "—",
                            "Capital After":  f"₹{cap_after:,.0f}" if cap_after else "—",
                            "Status":         "✅ Closed" if is_closed else "🟢 Open",
                        })
                    elif trade_type == "EXIT":
                        result = "✅ Profit" if pnl > 0 else ("❌ Loss" if pnl < 0 else "➖ B/E")
                        reason = r.get("reason", "").split()[0] if r.get("reason") else "—"
                        exit_px = float(r.get("price", 0))
                        journal.append({
                            "Time":           pd.to_datetime(r["time"], format="ISO8601").strftime("%d %b %H:%M"),
                            "Type":           "🏁 EXIT",
                            "Symbol":         symbol,
                            "Qty":            r.get("qty"),
                            "Entry Price":    f"₹{float(r.get('entry_price',0)):,.2f}" if r.get('entry_price') else "—",
                            "Exit Price":     f"₹{exit_px:,.2f}",
                            "Live Price":     f"₹{live_price:,.2f}" if live_price else "—",
                            "ATM Strike":     f"₹{live_atm:,.0f}" if live_atm else "—",
                            "Call Price(CE)": f"₹{ce_ltp:,.2f}" if ce_ltp else "—",
                            "Put Price(PE)":  f"₹{pe_ltp:,.2f}" if pe_ltp else "—",
                            "Options Action": opt_action,
                            "Expiry":         expiry,
                            "Stop":           "—",
                            "Target":         "—",
                            "Unrealized":     "—",
                            "Realised P&L":   f"₹{pnl:+,.0f}",
                            "Capital After":  f"₹{cap_after:,.0f}" if cap_after else "—",
                            "Status":         f"✅ Closed · {result} · {reason}",
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

                styled = df_journal.style.map(
                    _style_cell, subset=["Realised P&L", "Unrealized"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
                st.caption(
                    f"Prices at **{fetched_at}** IST · "
                    f"ATM Strike = nearest 50-pt exercise price · "
                    f"Call Price(CE) & Put Price(PE) = option premiums at that strike · "
                    f"click 🔄 Refresh for latest"
                )

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

    # ── Options Market Data ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Options Market Data")
    st.caption("Live ATM call & put prices for symbols in the journal. Refreshes with the page.")

    opts_file = PROJECT / "data/trades_log.jsonl"
    if opts_file.exists():
        try:
            rows      = [json.loads(l) for l in opts_file.read_text().splitlines() if l.strip()]
            syms_seen = {r["symbol"] for r in rows if r.get("symbol")}
            live_px   = _fetch_live_prices(tuple(syms_seen)) if syms_seen else {}

            opts_rows = []
            for sym in sorted(syms_seen):
                cur_price = live_px.get(sym)
                if not cur_price:
                    continue
                opts = _fetch_atm_options(sym, cur_price)
                short_sym = sym.replace("NSE:","").replace("-INDEX","").replace("-EQ","")

                # Direction of last signal for this symbol
                sig_file = PROJECT / "data/signals_log.jsonl"
                last_dir = 0
                if sig_file.exists():
                    sigs = [json.loads(l) for l in sig_file.read_text().splitlines() if l.strip()
                            and json.loads(l).get("symbol") == sym]
                    if sigs:
                        last_dir = int(sigs[-1].get("direction", 0))

                suggested = "BUY CE 📈" if last_dir == 1 else ("BUY PE 📉" if last_dir == -1 else "—")

                opts_rows.append({
                    "Symbol":       short_sym,
                    "Spot Price":   f"₹{cur_price:,.2f}",
                    "ATM Strike":   f"₹{opts['atm_strike']:,.0f}",
                    "Expiry":       opts["expiry"],
                    "CE (Call) ₹":  f"₹{opts['ce_ltp']:,.2f}" if opts["ce_ltp"] else "—",
                    "PE (Put) ₹":   f"₹{opts['pe_ltp']:,.2f}" if opts["pe_ltp"] else "—",
                    "Options Signal": suggested,
                    "CE Symbol":    opts["ce_sym"],
                    "PE Symbol":    opts["pe_sym"],
                })

            if opts_rows:
                df_opts = pd.DataFrame(opts_rows)
                st.dataframe(df_opts, use_container_width=True, hide_index=True)
                st.caption(f"Prices as of **{live_px.get('_at','—')}** IST · ATM = nearest 50-pt strike · CE = Call · PE = Put")
            else:
                st.info("Start paper trading to see options data here.")
        except Exception as e:
            st.warning(f"Options data unavailable: {e}")
    else:
        st.info("No trades yet — options data will appear once paper trading starts.")

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
