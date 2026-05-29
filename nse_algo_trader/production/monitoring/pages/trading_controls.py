"""Trading Controls — start/stop paper trading, live P&L, circuit breakers.

Uses subprocess + PID file so the trading process survives dashboard refreshes.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yaml

IST        = ZoneInfo("Asia/Kolkata")
PID_FILE   = Path("data/.paper_trader.pid")
STATE_FILE = Path("data/paper_trader_state.json")
LOG_DIR    = Path("logs")
PROJECT    = Path(__file__).parents[3]          # nse_algo_trader/
PYTHON     = sys.executable


# ── Process helpers ────────────────────────────────────────────────────────────

def _is_running() -> bool:
    """True if the paper trading process is alive."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)          # signal 0 = existence check, no actual signal
        return True
    except (ProcessLookupError, ValueError, OSError):
        PID_FILE.unlink(missing_ok=True)
        return False


def _start(symbols: list[str], interval: int, capital: float) -> bool:
    """Launch run_paper_trading.py as a detached subprocess. Returns True on success."""
    cmd = [
        PYTHON, str(PROJECT / "run_paper_trading.py"),
        "--symbols", *symbols,
        "--interval", str(interval),
        "--capital", str(int(capital)),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,   # detach from Streamlit's process group
        )
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(proc.pid))
        _write_state({**_read_state(), "running": True,
                      "symbols": symbols, "interval": interval, "capital": capital,
                      "pid": proc.pid})
        return True
    except Exception as e:
        st.error(f"Failed to start: {e}")
        return False


def _stop() -> None:
    """Send SIGTERM to the paper trading process."""
    if not PID_FILE.exists():
        return
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        # Force kill if still alive
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except (ValueError, ProcessLookupError, OSError):
        pass
    PID_FILE.unlink(missing_ok=True)
    _write_state({**_read_state(), "running": False})


def _read_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"running": False, "symbols": [], "interval": 15, "capital": 1_000_000,
            "equity": 0, "total_pnl": 0, "total_pnl_pct": 0,
            "trades": 0, "win_rate": 0, "profit_factor": 0,
            "daily_pnl_pct": 0, "max_dd_pct": 0,
            "positions": [], "circuit_broken": False, "circuit_reason": "",
            "last_update": "—", "signals_today": 0, "market_open": False}


def _write_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, default=str))


def _tail_log(n: int = 12) -> list[str]:
    """Return last N lines from today's paper trading log."""
    today = datetime.now(IST).strftime("%Y-%m-%d")
    log = LOG_DIR / f"paper_trading_{today}.log"
    if not log.exists():
        return []
    lines = log.read_text(errors="replace").splitlines()
    return lines[-n:]


# ── Top-up helpers ────────────────────────────────────────────────────────────

TOPUP_HISTORY_FILE = Path("data/.topup_history.json")


def _load_topup_history() -> list[dict]:
    if TOPUP_HISTORY_FILE.exists():
        try:
            return json.loads(TOPUP_HISTORY_FILE.read_text())
        except Exception:
            pass
    return []


def _save_topup_event(amount: float, reason: str, capital_before: float, capital_after: float):
    history = _load_topup_history()
    history.append({
        "time":           datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "amount":         amount,
        "reason":         reason,
        "capital_before": capital_before,
        "capital_after":  capital_after,
    })
    TOPUP_HISTORY_FILE.write_text(json.dumps(history, indent=2))


def _apply_topup(state: dict, add_amount: float, reason: str, running: bool) -> None:
    """Stop session, increase capital, restart with same symbols/interval."""
    old_capital = float(state.get("capital", 1_000_000))
    new_capital = old_capital + add_amount
    _save_topup_event(add_amount, reason, old_capital, new_capital)

    symbols  = state.get("symbols",  ["NSE:HDFCBANK-EQ"])
    interval = state.get("interval", 15)

    if running:
        _stop()
        time.sleep(2)

    _write_state({**_read_state(), "capital": new_capital, "equity": new_capital,
                  "total_pnl": 0, "total_pnl_pct": 0, "trades": 0,
                  "daily_pnl_pct": 0, "max_dd_pct": 0, "positions": []})
    _start(symbols, interval, new_capital)


def _render_topup(state: dict, running: bool) -> None:
    """Demo capital top-up panel."""
    capital  = float(state.get("capital", 1_000_000))
    equity   = float(state.get("equity", capital))
    loss_pct = (capital - equity) / capital * 100 if capital > 0 else 0

    st.subheader("💰 Demo Capital")

    c1, c2, c3 = st.columns(3)
    c1.metric("Starting Capital", f"₹{capital:,.0f}")
    c2.metric("Current Equity",   f"₹{equity:,.0f}",
              f"{(equity-capital)/capital*100:+.2f}%")
    c3.metric("Capital Remaining", f"{equity/capital*100:.1f}%",
              "⚠️ Low" if equity < capital * 0.5 else ("✓ Healthy" if equity >= capital * 0.9 else None))

    # Warn if more than 20% lost
    if loss_pct > 20:
        st.warning(f"⚠️  You've lost **{loss_pct:.1f}%** (₹{capital-equity:,.0f}) of your demo capital. Add funds below to continue trading.")

    with st.expander("➕  Add Demo Funds", expanded=(loss_pct > 20)):
        st.caption("This restarts the paper trading session with extra capital. Your trade history is preserved.")

        preset_col, custom_col = st.columns([2, 1])
        with preset_col:
            preset = st.radio(
                "Quick amounts",
                ["₹1,00,000", "₹5,00,000", "₹10,00,000", "Custom"],
                horizontal=True,
                label_visibility="collapsed",
            )
        with custom_col:
            custom_amt = st.number_input(
                "Custom amount (₹)",
                min_value=10_000, max_value=50_000_000,
                value=1_000_000, step=100_000,
                label_visibility="collapsed",
            )

        amount_map = {"₹1,00,000": 100_000, "₹5,00,000": 500_000,
                      "₹10,00,000": 1_000_000, "Custom": custom_amt}
        add_amount = amount_map[preset]

        reason = st.text_input("Reason (optional)", placeholder="e.g. testing new strategy, capital ran out")

        st.markdown(f"**After top-up:** ₹{capital:,.0f} + ₹{add_amount:,.0f} = **₹{capital+add_amount:,.0f}**")

        if st.button(f"💰  Add ₹{add_amount:,.0f} Demo Capital", type="primary", use_container_width=True):
            with st.spinner("Applying top-up and restarting session..."):
                _apply_topup(state, add_amount, reason or "Manual top-up", running)
                time.sleep(3)
            st.success(f"✅  ₹{add_amount:,.0f} added. New capital: ₹{capital+add_amount:,.0f}. Session restarted.")
            time.sleep(1)
            st.rerun()

    # Full reset option
    with st.expander("🔄  Reset to Fresh Start"):
        st.caption("Wipe all P&L history and restart with a clean capital amount.")
        reset_capital = st.number_input(
            "Fresh capital (₹)", min_value=100_000, max_value=50_000_000,
            value=int(capital), step=100_000, key="reset_capital",
        )
        if st.button("🔄  Reset & Restart Fresh", type="secondary", use_container_width=True):
            with st.spinner("Resetting..."):
                if running:
                    _stop(); time.sleep(2)
                symbols  = state.get("symbols", ["NSE:HDFCBANK-EQ"])
                interval = state.get("interval", 15)
                _write_state({**_read_state(), "capital": reset_capital, "equity": reset_capital,
                              "total_pnl": 0, "total_pnl_pct": 0, "trades": 0,
                              "daily_pnl_pct": 0, "max_dd_pct": 0, "positions": [],
                              "signals_today": 0})
                _start(symbols, interval, float(reset_capital))
                time.sleep(3)
            st.success(f"✅  Fresh start with ₹{reset_capital:,.0f}. Good luck!")
            st.rerun()

    # Top-up history
    history = _load_topup_history()
    if history:
        with st.expander(f"📋  Top-up History ({len(history)} events)"):
            hist_df = pd.DataFrame(reversed(history))
            hist_df["amount"]         = hist_df["amount"].apply(lambda x: f"₹{x:,.0f}")
            hist_df["capital_before"] = hist_df["capital_before"].apply(lambda x: f"₹{x:,.0f}")
            hist_df["capital_after"]  = hist_df["capital_after"].apply(lambda x: f"₹{x:,.0f}")
            hist_df.columns = ["Time", "Amount Added", "Reason", "Capital Before", "Capital After"]
            st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ── Page ───────────────────────────────────────────────────────────────────────

def render():
    st.title("🎮 Trading Controls")

    running = _is_running()
    state   = _read_state()

    # Keep state file in sync with actual process status
    if state.get("running") and not running:
        _write_state({**state, "running": False})
        state["running"] = False

    now = datetime.now(IST)
    market_open = (9 * 60 + 15) <= (now.hour * 60 + now.minute) <= (15 * 60 + 30) and now.weekday() < 5

    # ── Status banner ─────────────────────────────────────────────────────────
    if running:
        syms = ", ".join(s.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
                         for s in state.get("symbols", []))
        st.success(f"🟢  **PAPER TRADING ACTIVE** — {syms} · {state.get('interval')}min · last update {state.get('last_update', '—')}")
    elif state.get("circuit_broken"):
        st.error(f"🔴  **CIRCUIT BREAKER TRIPPED** — {state.get('circuit_reason', '')} — manual reset required")
    else:
        st.warning("🔴  **PAPER TRADING STOPPED**")

    st.divider()

    # ── Big START / STOP button ────────────────────────────────────────────────
    if not running:
        # ── Configuration form ─────────────────────────────────────────────────
        with st.form("start_form", border=True):
            st.markdown("### ▶️  Start Paper Trading")

            col_sym, col_right = st.columns([3, 2])
            with col_sym:
                symbols = st.multiselect(
                    "Symbols to trade",
                    options=[
                        "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:RELIANCE-EQ",
                        "NSE:TCS-EQ", "NSE:SBIN-EQ", "NSE:AXISBANK-EQ",
                        "NSE:INFY-EQ", "NSE:KOTAKBANK-EQ", "NSE:LT-EQ",
                        "NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX",
                    ],
                    default=state.get("symbols") or ["NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:RELIANCE-EQ"],
                )
                interval = st.select_slider(
                    "Bar interval",
                    options=[5, 15],
                    value=state.get("interval", 15),
                    format_func=lambda x: f"{x} min",
                )

            with col_right:
                capital = st.number_input(
                    "Paper capital (₹)",
                    min_value=100_000, max_value=50_000_000,
                    value=int(state.get("capital", 1_000_000)),
                    step=100_000,
                    format="%d",
                )
                st.caption(f"Max position : ₹{capital * 0.10:,.0f}")
                st.caption(f"Daily stop   : ₹{capital * 0.02:,.0f}  (–2%)")
                st.caption(f"DD pause     : ₹{capital * 0.05:,.0f}  (–5%)")

            if not market_open:
                st.info(f"⏰  Market is {'closed — opens 09:15 IST' if now.hour < 9 or (now.hour == 9 and now.minute < 15) else 'closed for today'}. "
                        f"Trading will begin automatically at 09:15.")

            start_btn = st.form_submit_button(
                "▶️  START PAPER TRADING",
                type="primary",
                use_container_width=True,
            )

        if start_btn:
            if not symbols:
                st.error("Select at least one symbol.")
            else:
                with st.spinner("Starting paper trading session..."):
                    ok = _start(symbols, interval, float(capital))
                if ok:
                    time.sleep(2)
                    st.rerun()

    else:
        # ── Live stats while running ────────────────────────────────────────────
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Capital",     f"₹{state.get('capital', 0):,.0f}")
        p2.metric("Total P&L",   f"₹{state.get('total_pnl', 0):+,.0f}",
                  f"{state.get('total_pnl_pct', 0):+.2f}%")
        p3.metric("Daily P&L",   f"{state.get('daily_pnl_pct', 0):+.2f}%",
                  "⚠️ near stop" if abs(state.get("daily_pnl_pct", 0)) > 1.5 else None)
        p4.metric("Win Rate",    f"{state.get('win_rate', 0):.1f}%")
        p5.metric("Trades",      state.get("trades", 0))

        st.divider()

        # ── STOP button ────────────────────────────────────────────────────────
        col_stop, col_info = st.columns([1, 2])
        with col_stop:
            if st.button("⏹️  STOP PAPER TRADING", type="primary", use_container_width=True):
                with st.spinner("Stopping..."):
                    _stop()
                    time.sleep(2)
                st.rerun()
        with col_info:
            syms = " · ".join(s.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
                              for s in state.get("symbols", []))
            st.info(f"Trading: **{syms}** on **{state.get('interval')}min** bars")

    st.divider()

    # ── Demo Capital / Top-Up ──────────────────────────────────────────────────
    _render_topup(state, running)

    st.divider()

    # ── Open positions ─────────────────────────────────────────────────────────
    positions = state.get("positions", [])
    st.subheader(f"Open Positions ({len(positions)})")
    if positions:
        pos_df = pd.DataFrame(positions)
        pos_df["direction"] = pos_df["direction"].map({"LONG": "🟢 LONG", "SHORT": "🔴 SHORT"})
        pos_df["entry"] = pos_df["entry"].apply(lambda x: f"₹{x:,.2f}")
        pos_df["mtm"]   = pos_df["mtm"].apply(lambda x: f"₹{x:+,.0f}")
        pos_df.columns  = ["Symbol", "Direction", "Entry Price", "Qty", "MTM P&L"]
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No open positions.")

    st.divider()

    # ── Live log tail ──────────────────────────────────────────────────────────
    st.subheader("Live Log")
    log_lines = _tail_log(15)
    if log_lines:
        COLORS = {"CRITICAL": "#ff4c6a", "ERROR": "#ff4c6a", "WARNING": "#ffd166",
                  "SUCCESS": "#00c896", "INFO": "#8892a4", "DEBUG": "#555"}
        html_lines = []
        for line in log_lines:
            color = "#8892a4"
            for level, c in COLORS.items():
                if f"| {level}" in line or f"|{level}" in line:
                    color = c
                    break
            safe = line.replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(f'<span style="color:{color};font-family:monospace;font-size:12px">{safe}</span>')
        st.markdown("<br>".join(html_lines), unsafe_allow_html=True)
    else:
        st.caption("No logs yet for today.")

    st.divider()

    # ── Circuit breaker panel ──────────────────────────────────────────────────
    st.subheader("⚡ Circuit Breaker")
    if state.get("circuit_broken"):
        st.error(f"**TRIPPED** — {state.get('circuit_reason', 'Unknown')}")
        st.warning("Investigate the cause before resetting. All trading is halted.")
        if st.button("🔄  Reset Circuit Breaker", type="secondary"):
            _write_state({**_read_state(), "circuit_broken": False, "circuit_reason": ""})
            st.success("Reset. You can now restart trading.")
            st.rerun()
    else:
        st.success("✅  All circuit breakers normal")
        try:
            cb = yaml.safe_load((PROJECT / "configs/risk.yaml").read_text()).get("circuit_breakers", {})
            c1, c2, c3 = st.columns(3)
            c1.caption(f"WS stale → pause: **{cb.get('websocket_stale_sec', 30)}s**")
            c2.caption(f"VIX spike halt: **{cb.get('vix_spike_pct', 30)}%** in 15min")
            c3.caption(f"Max rejections: **{cb.get('max_order_rejections', 3)}** / 10min")
        except Exception:
            pass

    st.divider()

    # ── Strategy quick-tune ────────────────────────────────────────────────────
    with st.expander("🔧 Quick Strategy Tuning (takes effect on next session start)"):
        try:
            cfg = yaml.safe_load((PROJECT / "configs/strategy.yaml").read_text())
            m = cfg.get("momentum", {}); v = cfg.get("vix", {}); r = cfg.get("regime", {})

            qc1, qc2, qc3 = st.columns(3)
            conf  = qc1.slider("Min confidence", 0.50, 0.95, 0.60, 0.05)
            adx   = qc2.slider("Min ADX",        15,   40,   25,   5)
            vix_m = qc3.slider("Max VIX",        15,   35,   25,   5)

            rr = m.get("target_atr_multiplier", 3.0) / m.get("stop_atr_multiplier", 2.0)
            st.caption(f"Current R:R = **{rr:.1f}:1** (stop {m.get('stop_atr_multiplier')}×ATR → target {m.get('target_atr_multiplier')}×ATR)")

            if st.button("💾  Save Quick Tune"):
                cfg.setdefault("momentum", {})["confidence_threshold"] = conf
                cfg.setdefault("regime",   {})["adx_trending_threshold"] = adx
                cfg.setdefault("vix",      {})["high"] = vix_m
                (PROJECT / "configs/strategy.yaml").write_text(
                    yaml.dump(cfg, default_flow_style=False, sort_keys=False))
                st.success("Saved — restart session to apply.")
        except Exception as e:
            st.info(f"Could not load strategy config: {e}")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if running:
        if st.toggle("Auto-refresh every 15s", value=False, key="auto_refresh"):
            time.sleep(15)
            st.rerun()
