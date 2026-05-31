"""Strategy Info page — documentation, live parameter editing, and Claude request form."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import yaml

CONFIG_PATH   = Path("configs/strategy.yaml")
REQUESTS_FILE = Path("data/strategy_requests.jsonl")
IST           = ZoneInfo("Asia/Kolkata")


def _load_cfg() -> dict:
    try:
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except Exception:
        return {}


def _save_cfg(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


def _save_request(request: str, params_changed: dict) -> None:
    REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time":           datetime.now(IST).isoformat(),
        "request":        request.strip(),
        "params_changed": params_changed,
        "status":         "pending",
    }
    with REQUESTS_FILE.open("a") as f:
        f.write(json.dumps(row) + "\n")


def render() -> None:
    st.title("📋 Phase 1 — Momentum Strategy")
    st.caption("Review rules, tune parameters, or send a modification request to Claude.")

    cfg = _load_cfg()
    m   = cfg.get("momentum", {})
    r   = cfg.get("regime",   {})
    s   = cfg.get("session",  {})
    e   = cfg.get("entry",    {})
    v   = cfg.get("vix",      {})

    # ── 1. Strategy overview ──────────────────────────────────────────────────
    with st.expander("📖 Strategy Overview", expanded=True):
        st.markdown("""
**EMA 9/21 Crossover with Trend & Quality Filters**

The momentum strategy enters trades when a short-term EMA crosses a
medium-term EMA *and* several quality gates are all green. Position sizing
uses a fixed 5 % of capital per trade with ATR-based stop and target.

| Step | Condition | Purpose |
|------|-----------|---------|
| 1 | EMA 9 crosses EMA 21 | Trend direction change |
| 2 | ADX ≥ threshold | Market is trending, not choppy |
| 3 | +DI > −DI (long) / −DI > +DI (short) | Directional confirmation |
| 4 | Volume ratio ≥ 1.2× | Institutional participation |
| 5 | RSI in healthy range | Not entering at extremes |
| 6 | Session window 09:45–15:10 | Skip opening noise & late entries |
| 7 | Confidence ≥ threshold | Composite signal strength gate |

**Exit rules**
- Stop-loss: `stop_atr_multiplier × ATR` below entry
- Take-profit: `target_atr_multiplier × ATR` above entry
- EOD forced exit at 15:25 IST
- Signal reversal triggers early exit
        """)

    st.divider()

    # ── 2. Live parameters ────────────────────────────────────────────────────
    st.subheader("⚙️ Strategy Parameters")
    st.caption("Edit and click **Apply Changes** to save directly to strategy.yaml.")

    changed: dict = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Signal Filters**")
        adx = st.number_input(
            "ADX trending threshold",
            min_value=10.0, max_value=50.0, step=1.0,
            value=float(r.get("adx_trending_threshold", 25)),
            help="ADX must exceed this to confirm a trending market (was 25, lowered to 20 for more signals)",
        )
        vol = st.number_input(
            "Volume filter ratio",
            min_value=0.5, max_value=3.0, step=0.1,
            value=float(m.get("volume_filter_ratio", 1.2)),
            help="Volume must be this multiple of the 20-bar average to confirm entry",
        )
        rsi_long_min = st.number_input(
            "RSI long floor", min_value=20.0, max_value=60.0, step=1.0,
            value=float(e.get("rsi_long_min", 45)),
            help="RSI must be above this to go long (avoids oversold fakeouts)",
        )
        rsi_long_max = st.number_input(
            "RSI long ceiling", min_value=50.0, max_value=90.0, step=1.0,
            value=float(e.get("rsi_long_max", 70)),
            help="RSI must be below this to go long (avoids overbought entries)",
        )
        rsi_short_min = st.number_input(
            "RSI short floor", min_value=10.0, max_value=50.0, step=1.0,
            value=float(e.get("rsi_short_min", 30)),
            help="RSI must be above this to go short",
        )
        rsi_short_max = st.number_input(
            "RSI short ceiling", min_value=40.0, max_value=80.0, step=1.0,
            value=float(e.get("rsi_short_max", 55)),
            help="RSI must be below this to go short (avoids oversold entries)",
        )

    with col2:
        st.markdown("**Risk & Sizing**")
        stop_mult = st.number_input(
            "Stop-loss ATR multiplier",
            min_value=0.5, max_value=5.0, step=0.5,
            value=float(m.get("stop_atr_multiplier", 2.0)),
            help="Stop = entry ± (this × ATR). Larger = wider stop, fewer stop-outs",
        )
        target_mult = st.number_input(
            "Take-profit ATR multiplier",
            min_value=1.0, max_value=8.0, step=0.5,
            value=float(m.get("target_atr_multiplier", 3.0)),
            help="Target = entry ± (this × ATR). R:R = target_mult / stop_mult",
        )
        confidence = st.number_input(
            "Min confidence threshold",
            min_value=0.4, max_value=0.9, step=0.05,
            value=float(e.get("min_confidence", cfg.get("signal_confidence_threshold", 0.55))),
            help="Composite signal score (0–1). Higher = fewer but higher quality trades",
        )
        sess_start = st.text_input(
            "Session start (HH:MM)", value=s.get("start", "09:45"),
            help="No entries before this time (IST)",
        )
        sess_end = st.text_input(
            "Session end (HH:MM)", value=s.get("end", "15:10"),
            help="No new entries after this time (IST)",
        )
        vix_max = st.number_input(
            "VIX max (entry gate)",
            min_value=15.0, max_value=50.0, step=1.0,
            value=float(v.get("high", 25)),
            help="Block new entries when India VIX exceeds this level",
        )

    rr = round(target_mult / stop_mult, 2)
    st.info(f"**Current R:R = {rr}:1** — breakeven win rate = {round(1/(1+rr)*100,1)}%")

    if st.button("✅ Apply Changes", type="primary"):
        cfg.setdefault("regime",  {})["adx_trending_threshold"] = adx
        cfg.setdefault("momentum", {})["volume_filter_ratio"]   = vol
        cfg.setdefault("momentum", {})["stop_atr_multiplier"]   = stop_mult
        cfg.setdefault("momentum", {})["target_atr_multiplier"] = target_mult
        cfg.setdefault("entry",   {})["rsi_long_min"]           = rsi_long_min
        cfg.setdefault("entry",   {})["rsi_long_max"]           = rsi_long_max
        cfg.setdefault("entry",   {})["rsi_short_min"]          = rsi_short_min
        cfg.setdefault("entry",   {})["rsi_short_max"]          = rsi_short_max
        cfg.setdefault("entry",   {})["min_confidence"]         = confidence
        cfg["signal_confidence_threshold"]                       = confidence
        cfg.setdefault("session", {})["start"]                  = sess_start
        cfg.setdefault("session", {})["end"]                    = sess_end
        cfg.setdefault("vix",     {})["high"]                   = vix_max

        changed = {
            "adx_trending_threshold": adx, "volume_filter_ratio": vol,
            "stop_atr_multiplier": stop_mult, "target_atr_multiplier": target_mult,
            "rsi_long_min": rsi_long_min, "rsi_long_max": rsi_long_max,
            "rsi_short_min": rsi_short_min, "rsi_short_max": rsi_short_max,
            "min_confidence": confidence, "session_start": sess_start,
            "session_end": sess_end, "vix_max": vix_max,
        }
        try:
            _save_cfg(cfg)
            st.success("Parameters saved to `configs/strategy.yaml`. Restart paper trading to apply.")
            st.json(changed)
        except Exception as ex:
            st.error(f"Save failed: {ex}")

    st.divider()

    # ── 3. Request to Claude ──────────────────────────────────────────────────
    st.subheader("💬 Request a Strategy Change")
    st.caption(
        "Describe what you'd like changed in plain English. "
        "Claude Code will read this request, implement the change, and restart the strategy."
    )

    request_text = st.text_area(
        "Your request",
        placeholder=(
            "e.g. 'Tighten the stop to 1.5 ATR and widen the target to 4 ATR to improve R:R'\n"
            "e.g. 'Add a higher-timeframe trend filter — only go long if 1h EMA9 > EMA21'\n"
            "e.g. 'Lower ADX threshold to 18 and see if that improves trade count'"
        ),
        height=120,
    )

    priority = st.selectbox("Priority", ["Normal", "High", "Low"])

    if st.button("📨 Submit Request to Claude", type="secondary"):
        if not request_text.strip():
            st.warning("Please describe the change you'd like made.")
        else:
            _save_request(request_text, changed)
            st.success(
                "Request saved! Open Claude Code and say "
                "**'check pending strategy requests'** to apply it."
            )
            st.code(f"Request logged at: {REQUESTS_FILE}", language="text")

    # ── 4. Pending requests log ───────────────────────────────────────────────
    if REQUESTS_FILE.exists():
        lines = REQUESTS_FILE.read_text().strip().splitlines()
        pending = []
        for line in lines:
            try:
                row = json.loads(line)
                if row.get("status") == "pending":
                    pending.append(row)
            except Exception:
                pass

        if pending:
            st.divider()
            st.subheader(f"🕐 Pending Requests ({len(pending)})")
            for req in reversed(pending[-5:]):
                with st.expander(f"{req['time'][:16]}  —  {req['request'][:60]}..."):
                    st.write(req["request"])
                    if req.get("params_changed"):
                        st.json(req["params_changed"])
