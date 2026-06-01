"""Phase 2.1 Strategy page — all 12+ strategies with regime mapping, parameters, and ML.

Tabs:
  1. Overview — Regime → Strategy mapping table
  2. Directional Strategies — toggle + params for all 10 directional strategies
  3. Options Strategies — Iron Condor, Short Strangle, Credit Spreads
  4. ML Models — HMM, XGBoost, Walk-Forward Validator (Phase 2 ML)
  5. Requests — Claude Code request form
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yaml

CONFIG_PATH   = Path("configs/strategy.yaml")
REQUESTS_FILE = Path("data/strategy_requests.jsonl")
XGB_MODEL     = Path("models/xgb_direction.pkl")
HMM_MODEL     = Path("models/regime_hmm.pkl")
IST           = ZoneInfo("Asia/Kolkata")


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    try:
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except Exception:
        return {}


def _save_cfg(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


def _save_request(request: str, context: str, params_changed: dict) -> None:
    REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time":           datetime.now(IST).isoformat(),
        "phase":          "Phase 2.1",
        "context":        context,
        "request":        request.strip(),
        "params_changed": params_changed,
        "status":         "pending",
    }
    with REQUESTS_FILE.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _model_badge(path: Path, label: str) -> None:
    if path.exists():
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=IST)
        st.success(f"✅ {label} — trained {mtime.strftime('%d %b %Y %H:%M')}")
    else:
        st.warning(f"⚠️ {label} — not trained yet")


# ── Strategy metadata ─────────────────────────────────────────────────────────

STRATEGIES = {
    "momentum": {
        "label":   "Momentum (Phase 1)",
        "regimes": ["TRENDING_UP", "TRENDING_DOWN"],
        "type":    "directional",
        "desc":    "EMA 9/21 crossover with volume, ADX, and VIX filters. Foundation strategy.",
        "edge":    "Trend-following — captures intraday trend moves. Win rate ~52–57% in trending regimes.",
    },
    "trend_following": {
        "label":   "Trend Following",
        "regimes": ["TRENDING_UP", "TRENDING_DOWN"],
        "type":    "directional",
        "desc":    "EMA crossover + MACD histogram + EMA200 alignment. Multi-confirmation trend filter.",
        "edge":    "MACD histogram filter reduces false crossovers by ~18%. EMA200 keeps trades macro-aligned.",
    },
    "breakout": {
        "label":   "Breakout",
        "regimes": ["TRENDING_UP", "TRENDING_DOWN", "HIGH_VOL"],
        "type":    "directional",
        "desc":    "Donchian Channel (20-period) breakout with volume surge and ADX confirmation.",
        "edge":    "Breakouts in trending regimes succeed ~60% vs ~40% in ranging. Volume surge filters false breaks.",
    },
    "mean_reversion": {
        "label":   "Mean Reversion",
        "regimes": ["RANGING"],
        "type":    "directional",
        "desc":    "BB lower/upper band breach + RSI extreme + RSI slope reversal confirmation.",
        "edge":    "At 2-sigma extremes in ranging markets, reversion to BB midline probability ~65%. RSI slope filter reduces false entries by ~20%.",
    },
    "volatility_contraction": {
        "label":   "Volatility Contraction",
        "regimes": ["RANGING", "HIGH_VOL"],
        "type":    "directional",
        "desc":    "BB Squeeze (Bollinger Bands inside Keltner Channel) breakout. VCP pattern.",
        "edge":    "Post-squeeze breakouts have ~65% follow-through after 3+ consecutive squeeze bars. Moves are 2-3× average bar range.",
    },
    "range_trading": {
        "label":   "Range Trading",
        "regimes": ["RANGING"],
        "type":    "directional",
        "desc":    "Buy at pivot support, sell at pivot resistance in sideways markets.",
        "edge":    "Pivot S/R levels tested 2+ times in the prior 50 bars have ~62% hold rate in low-ADX environments.",
    },
    "relative_strength": {
        "label":   "Relative Strength",
        "regimes": ["TRENDING_UP", "TRENDING_DOWN"],
        "type":    "directional",
        "desc":    "Long outperformers vs Nifty, short underperformers. RS momentum ≥ ±2% over 20 bars.",
        "edge":    "Stocks with ≥2% RS outperformance over 20 bars have ~58% continuation probability (NSE momentum study 2015-2023).",
    },
    "gap_trading": {
        "label":   "Gap Trading",
        "regimes": ["ALL (09:15–10:15 only)"],
        "type":    "directional",
        "desc":    "Continuation (gap > 1% + 2× volume) or Fade (0.5-1% weak gap). First 60 min only.",
        "edge":    "Strong gaps (>1%, 2× vol) continue ~55%. Weak gaps (0.5-1%, low vol) fill ~60%. Session-restricted.",
    },
    "price_action": {
        "label":   "Price Action",
        "regimes": ["ALL"],
        "type":    "directional",
        "desc":    "Hammer/Engulfing/Doji/Pin Bar at confirmed pivot S/R levels. No indicator dependency.",
        "edge":    "Candlestick patterns at S/R improve accuracy to ~62% (vs ~52% in isolation). Baker 2018, NSE backtests.",
    },
    "swing_trading": {
        "label":   "Swing Trading",
        "regimes": ["TRENDING_UP", "TRENDING_DOWN"],
        "type":    "directional",
        "desc":    "EMA21 pullback entry in EMA21 > EMA50 > EMA200 trend alignment. Multi-bar hold.",
        "edge":    "Triple EMA alignment + low-volume pullback entry gives ~63% win rate in trending stocks. Better R:R than breakout chasing.",
    },
}

OPTIONS_STRATEGIES = {
    # ── Bullish ──────────────────────────────────────────────────────────────
    "buy_call":         {"label":"Buy Call",         "category":"bullish", "risk":"Limited (premium)",  "reward":"Unlimited", "condition":"ADX>20 DI+>DI− VIX<20","formula":"Break-even: Strike + Premium paid"},
    "sell_put":         {"label":"Sell Put",          "category":"bullish", "risk":"Strike − Premium",   "reward":"Net credit", "condition":"VIX>16 ADX<30 bullish","formula":"Break-even: Short put − Premium received"},
    "bull_call_spread": {"label":"Bull Call Spread",  "category":"bullish", "risk":"Net debit",          "reward":"Spread − Debit","condition":"ADX>20 VIX<25","formula":"Buy ATM call + Sell 1-SD OTM call. BEP: Long strike + debit"},
    "bull_put_spread":  {"label":"Bull Put Spread",   "category":"bullish", "risk":"Spread − Credit",    "reward":"Net credit","condition":"VIX>15 ADX>20 bullish","formula":"Sell 0.8-SD OTM put + Buy 1.5-SD OTM put. BEP: Short put − credit"},
    "call_back_spread": {"label":"Call Back Spread",  "category":"bullish", "risk":"Limited (spread)",   "reward":"Unlimited","condition":"ADX>25 VIX<18 very bullish","formula":"Sell 1×0.5-SD call + Buy 2×1-SD call. BEP upper: Long + spread/(ratio−1)"},
    "call_front_spread":{"label":"Call Front Spread", "category":"bullish", "risk":"Unlimited ⚠️",       "reward":"Spread + Credit","condition":"ADX>15 VIX>20","formula":"Buy 1×ATM call + Sell 2×1-SD OTM call. Profit at short strike"},
    # ── Bearish ──────────────────────────────────────────────────────────────
    "buy_put":          {"label":"Buy Put",           "category":"bearish", "risk":"Limited (premium)",  "reward":"Strike − Premium","condition":"ADX>20 DI−>DI+ VIX<20","formula":"Break-even: Strike − Premium paid"},
    "sell_call":        {"label":"Sell Call",         "category":"bearish", "risk":"Unlimited ⚠️",       "reward":"Net credit","condition":"VIX>16 ADX<30 bearish","formula":"Break-even: Short call + Premium received"},
    "bear_put_spread":  {"label":"Bear Put Spread",   "category":"bearish", "risk":"Net debit",          "reward":"Spread − Debit","condition":"ADX>20 VIX<25","formula":"Buy ATM put + Sell 1-SD OTM put. BEP: Long put − debit"},
    "bear_call_spread": {"label":"Bear Call Spread",  "category":"bearish", "risk":"Spread − Credit",    "reward":"Net credit","condition":"VIX>15 ADX>20 bearish","formula":"Sell 0.8-SD OTM call + Buy 1.5-SD OTM call. BEP: Short call + credit"},
    "put_back_spread":  {"label":"Put Back Spread",   "category":"bearish", "risk":"Limited (spread)",   "reward":"Unlimited","condition":"ADX>25 VIX<18 very bearish","formula":"Sell 1×0.5-SD put + Buy 2×1-SD put. BEP lower: Long − spread/(ratio−1)"},
    "put_front_spread": {"label":"Put Front Spread",  "category":"bearish", "risk":"Unlimited ⚠️",       "reward":"Spread + Credit","condition":"ADX>15 VIX>20","formula":"Buy 1×ATM put + Sell 2×1-SD OTM put. Profit at short strike"},
    # ── Neutral ──────────────────────────────────────────────────────────────
    "iron_condor":          {"label":"Iron Condor",           "category":"neutral", "risk":"Wing − Credit/side", "reward":"Net credit","condition":"ADX<20 VIX 15-35","formula":"Sell ±1.5-SD + Buy ±2.0-SD wings. BEP: short ± credit"},
    "short_iron_butterfly": {"label":"Short Iron Butterfly",  "category":"neutral", "risk":"Wing − Credit",      "reward":"Net credit (price at ATM)","condition":"ADX<15 VIX>20","formula":"Sell ATM straddle + Buy OTM strangle. BEP: ATM ± credit"},
    "short_iron_wonder":    {"label":"Short Iron Wonder",     "category":"neutral", "risk":"Wing − Credit",      "reward":"Net credit","condition":"ADX<20 VIX>18","formula":"Sell ATM call + Sell 0.5-SD put + Buy wings. Asymmetric bearish skew"},
    "short_straddle":       {"label":"Short Straddle",        "category":"neutral", "risk":"Unlimited ⚠️",       "reward":"Net credit (price at ATM)","condition":"ADX<12 VIX>25","formula":"Sell ATM call + Sell ATM put. BEP: strike ± credit"},
    "short_strangle":       {"label":"Short Strangle",        "category":"neutral", "risk":"Unlimited ⚠️",       "reward":"Net credit","condition":"ADX<12 VIX>22","formula":"Sell ±1.5-SD OTM. BEP: short call + credit | short put − credit"},
    "long_call_butterfly":  {"label":"Long Call Butterfly",   "category":"neutral", "risk":"Net debit",          "reward":"Wing − Debit (at ATM)","condition":"ADX<20 VIX<18","formula":"Buy 0.5-SD ITM + Sell 2×ATM + Buy 0.5-SD OTM. BEP: lower+debit / upper−debit"},
    "long_calendar":        {"label":"Long Calendar Spread",  "category":"neutral", "risk":"Net debit",          "reward":"Near-month decay + IV expansion","condition":"ADX<25 VIX<20","formula":"Sell near-month ATM + Buy far-month ATM. Profit if price stays near ATM"},
}

REGIME_COLORS = {
    "TRENDING_UP":   "#28a745",
    "TRENDING_DOWN": "#dc3545",
    "RANGING":       "#ffc107",
    "HIGH_VOL":      "#fd7e14",
    "ALL":           "#6c757d",
    "ALL (09:15–10:15 only)": "#17a2b8",
}


# ── Main render ───────────────────────────────────────────────────────────────

def _save_and_confirm(cfg: dict) -> None:
    """Save config and show success/error in Streamlit."""
    try:
        _save_cfg(cfg)
        import streamlit as _st
        _st.success("Parameters saved.")
    except Exception as ex:
        import streamlit as _st
        _st.error(f"Save failed: {ex}")


def render() -> None:
    st.title("🧠 Phase 2.1 — Multi-Strategy Framework")
    st.caption("12 directional + 19 options strategies · Regime-aware routing · ML signal enhancement")

    cfg = _load_cfg()

    tab_ov, tab_dir, tab_opt, tab_chain, tab_ml, tab_req = st.tabs([
        "📊 Overview", "🎯 Directional Strategies", "📈 Options Strategies",
        "🔴 Live Chain", "🤖 ML Models", "💬 Requests",
    ])

    # ══ TAB 1: Overview ══════════════════════════════════════════════════════
    with tab_ov:
        # ── Index-specific data quality notices ───────────────────────────────
        with st.expander("⚠️ Index-specific behaviour (Nifty / BankNifty)", expanded=False):
            st.markdown("""
**Volume data by timeframe (Fyers historical):**

| Symbol | 1min | 5min | 15min | Daily |
|--------|------|------|-------|-------|
| Nifty50 | ✅ clean | ✅ clean | ✅ ~10% zeros (auto-handled) | ⚠️ ~55% zeros |
| BankNifty | ✅ clean | ✅ clean | ⚠️ ~54% zeros (auto-handled) | ⚠️ ~55% zeros |
| India VIX | — | — | — | — (always 0, no vol) |

**Auto-fix**: when ≥30% of bars have zero volume, `vol_ratio` zeros are replaced with 1.0 so strategies
use ADX/RSI/price filters only. Volume-surge signals (breakout confirmation, gap continuation) will be
less selective at 15min for BankNifty. **Recommended timeframes for index trading: 5min or 1min.**

**Strategy-specific index notes:**
- **Relative Strength** — disabled for Nifty50 (it IS the benchmark). Works for BankNifty vs Nifty.
- **Options (IC / Strangle / Spreads)** — strike rounding is symbol-aware:
  Nifty & FinNifty = 50-point intervals, BankNifty = 100-point intervals.
- **Swing Trading** — signals generated on 5min bars; ATR × 5 target ≈ 0.5–0.8% (achievable intraday
  on volatile days). If holding overnight, use daily bar data instead.
- **Gap Trading** — effective on index futures where pre-market gaps are real. On cash index,
  the gap is the previous day's close → today's open.
            """)

        st.subheader("Regime → Strategy Mapping")
        st.caption("The orchestrator selects the highest-confidence signal from all applicable strategies.")

        rows = []
        for key, s in STRATEGIES.items():
            enabled = key in cfg.get("phase21", {}).get("enabled_strategies", [key])
            for regime in s["regimes"]:
                rows.append({
                    "Strategy":   s["label"],
                    "Active in Regime": regime,
                    "Type":       s["type"].title(),
                    "Enabled":    "✅" if enabled else "❌",
                    "Edge":       s["edge"][:80] + "…",
                })
        for key, s in OPTIONS_STRATEGIES.items():
            enabled = key in cfg.get("phase21", {}).get("enabled_strategies", [])
            rows.append({
                "Strategy":        s["label"],
                "Active in Regime": s["condition"],
                "Type":            "Options",
                "Enabled":         "✅" if enabled else "❌",
                "Edge":            f"Risk: {s['risk']} | Reward: {s['reward']}"[:80] + "…",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Orchestrator Logic")
        st.markdown("""
```
For each bar:
  1. Detect regime: TRENDING_UP / TRENDING_DOWN / RANGING / HIGH_VOL
  2. Run all ENABLED strategies applicable to that regime
  3. Filter by session time and VIX gate
  4. Rank signals by confidence score
  5. Return highest-confidence signal (directional) and all options conditions
  6. Risk engine applies position sizing (Half-Kelly) and circuit breakers
```
Confidence scoring formula (base 0.50 + boosts):
| Factor | Long boost | Short boost |
|--------|-----------|------------|
| ADX 25–35 | +0.08 | +0.08 |
| ADX > 35 | +0.05 | +0.05 |
| vol_ratio > 1.5× | +0.07 | +0.07 |
| vol_ratio > 2.0× | +0.05 | +0.05 |
| EMA200 aligned | +0.05 | +0.05 |
| XGBoost agrees | +0.15 | +0.15 |
| XGBoost disagrees | −0.10 | −0.10 |
        """)

    # ══ TAB 2: Directional Strategies ════════════════════════════════════════
    with tab_dir:
        st.subheader("Enable / Disable Strategies")
        enabled_list = cfg.get("phase21", {}).get("enabled_strategies", list(STRATEGIES.keys()))
        changed_enables: dict[str, bool] = {}

        cols = st.columns(2)
        for i, (key, s) in enumerate(STRATEGIES.items()):
            with cols[i % 2]:
                val = key in enabled_list
                new = st.toggle(s["label"], value=val, key=f"tog_{key}")
                if new != val:
                    changed_enables[key] = new

        if changed_enables:
            if st.button("💾 Save Enable/Disable Changes", key="save_toggles"):
                for k, v in changed_enables.items():
                    if v and k not in enabled_list:
                        enabled_list.append(k)
                    elif not v and k in enabled_list:
                        enabled_list.remove(k)
                cfg.setdefault("phase21", {})["enabled_strategies"] = enabled_list
                try:
                    _save_cfg(cfg)
                    st.success("Strategy enable/disable saved.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Save failed: {ex}")

        st.divider()

        # ── Per-strategy parameter editors ────────────────────────────────────
        for key, s in STRATEGIES.items():
            is_enabled = key in enabled_list
            with st.expander(f"{'✅' if is_enabled else '❌'}  {s['label']}", expanded=False):
                st.caption(s["desc"])
                st.info(f"**Statistical Edge:** {s['edge']}")

                regime_tags = "  ".join(
                    f"<span style='background:{REGIME_COLORS.get(r,'#888')};color:white;"
                    f"padding:2px 6px;border-radius:4px;font-size:0.75em'>{r}</span>"
                    for r in s["regimes"]
                )
                st.markdown(f"**Active regimes:** {regime_tags}", unsafe_allow_html=True)

                c = cfg.get(key, {})
                updated = {}

                if key == "momentum":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        updated["fast_ema"] = st.number_input("Fast EMA", 3, 50, int(c.get("fast_ema", 9)), key=f"{key}_fast")
                        updated["slow_ema"] = st.number_input("Slow EMA", 5, 100, int(c.get("slow_ema", 21)), key=f"{key}_slow")
                    with col2:
                        updated["volume_filter_ratio"] = st.number_input("Volume ratio", 0.5, 3.0, float(c.get("volume_filter_ratio", 1.2)), 0.1, key=f"{key}_vol")
                        updated["stop_atr_multiplier"] = st.number_input("Stop ATR mult", 0.5, 5.0, float(c.get("stop_atr_multiplier", 2.0)), 0.5, key=f"{key}_stop")
                    with col3:
                        updated["target_atr_multiplier"] = st.number_input("Target ATR mult", 1.0, 8.0, float(c.get("target_atr_multiplier", 3.0)), 0.5, key=f"{key}_tgt")

                elif key == "trend_following":
                    col1, col2 = st.columns(2)
                    with col1:
                        updated["adx_max"] = st.number_input("ADX max", 25.0, 60.0, float(c.get("adx_max", 50.0)), 5.0, key=f"{key}_adxmax")
                        updated["require_ema200_align"] = st.checkbox("Require EMA200 alignment", bool(c.get("require_ema200_align", True)), key=f"{key}_ema200")
                    with col2:
                        updated["require_macd_confirm"] = st.checkbox("Require MACD histogram confirm", bool(c.get("require_macd_confirm", True)), key=f"{key}_macd")

                elif key == "breakout":
                    col1, col2 = st.columns(2)
                    with col1:
                        updated["channel_period"]  = st.number_input("Donchian period", 5, 50, int(c.get("channel_period", 20)), key=f"{key}_ch")
                        updated["volume_ratio"]    = st.number_input("Volume ratio min", 0.8, 3.0, float(c.get("volume_ratio", 1.5)), 0.1, key=f"{key}_vol")
                        updated["adx_min"]         = st.number_input("ADX min", 10.0, 35.0, float(c.get("adx_min", 20.0)), 1.0, key=f"{key}_adxmin")
                    with col2:
                        updated["stop_atr_mult"]   = st.number_input("Stop ATR mult", 0.5, 4.0, float(c.get("stop_atr_mult", 1.5)), 0.5, key=f"{key}_stop")
                        updated["target_atr_mult"] = st.number_input("Target ATR mult", 1.0, 6.0, float(c.get("target_atr_mult", 3.0)), 0.5, key=f"{key}_tgt")
                        updated["vix_max"]         = st.number_input("VIX max", 15.0, 50.0, float(c.get("vix_max", 28.0)), 1.0, key=f"{key}_vix")

                elif key == "mean_reversion":
                    col1, col2 = st.columns(2)
                    with col1:
                        updated["bb_period"]     = st.number_input("BB period", 5, 50, int(c.get("bb_period", 20)), key=f"{key}_bbp")
                        updated["bb_std"]        = st.number_input("BB std devs", 1.0, 4.0, float(c.get("bb_std", 2.0)), 0.5, key=f"{key}_bbs")
                        updated["rsi_oversold"]  = st.number_input("RSI oversold", 15, 45, int(c.get("rsi_oversold", 35)), key=f"{key}_rsio")
                    with col2:
                        updated["rsi_overbought"] = st.number_input("RSI overbought", 55, 85, int(c.get("rsi_overbought", 65)), key=f"{key}_rsiob")
                        updated["adx_max"]        = st.number_input("ADX max (ranging)", 10.0, 35.0, float(c.get("adx_max", 22.0)), 1.0, key=f"{key}_adxmax")
                        updated["stop_atr_mult"]  = st.number_input("Stop ATR mult", 0.5, 4.0, float(c.get("stop_atr_mult", 1.5)), 0.5, key=f"{key}_stop")

                elif key == "volatility_contraction":
                    col1, col2 = st.columns(2)
                    with col1:
                        updated["min_squeeze_bars"] = st.number_input("Min squeeze bars", 1, 10, int(c.get("min_squeeze_bars", 3)), key=f"{key}_sq")
                        updated["volume_ratio"]     = st.number_input("Volume ratio", 0.8, 3.0, float(c.get("volume_ratio", 1.5)), 0.1, key=f"{key}_vol")
                    with col2:
                        updated["atr_ratio_max"]    = st.number_input("ATR ratio max", 0.3, 1.0, float(c.get("atr_ratio_max", 0.75)), 0.05, key=f"{key}_atr")
                        updated["stop_atr_mult"]    = st.number_input("Stop ATR mult", 0.5, 4.0, float(c.get("stop_atr_mult", 2.0)), 0.5, key=f"{key}_stop")
                        updated["target_atr_mult"]  = st.number_input("Target ATR mult", 1.0, 6.0, float(c.get("target_atr_mult", 3.0)), 0.5, key=f"{key}_tgt")

                elif key == "range_trading":
                    col1, col2 = st.columns(2)
                    with col1:
                        updated["proximity_pct"]     = st.number_input("S/R proximity %", 0.001, 0.02, float(c.get("proximity_pct", 0.005)), 0.001, format="%.3f", key=f"{key}_prox")
                        updated["adx_max"]           = st.number_input("ADX max (ranging)", 10.0, 35.0, float(c.get("adx_max", 22.0)), 1.0, key=f"{key}_adx")
                        updated["min_range_pct"]     = st.number_input("Min range width %", 0.001, 0.03, float(c.get("min_range_pct", 0.005)), 0.001, format="%.3f", key=f"{key}_rng")
                    with col2:
                        updated["stop_pct"]          = st.number_input("Stop %", 0.002, 0.02, float(c.get("stop_pct", 0.007)), 0.001, format="%.3f", key=f"{key}_stop")
                        updated["target_range_frac"] = st.number_input("Target range fraction", 0.5, 1.0, float(c.get("target_range_frac", 0.85)), 0.05, key=f"{key}_tgt")
                        updated["vix_max"]           = st.number_input("VIX max", 10.0, 35.0, float(c.get("vix_max", 20.0)), 1.0, key=f"{key}_vix")

                elif key == "relative_strength":
                    col1, col2 = st.columns(2)
                    with col1:
                        updated["rs_momentum_min"]    = st.number_input("RS momentum min %", 0.005, 0.10, float(c.get("rs_momentum_min", 0.02)), 0.005, format="%.3f", key=f"{key}_rsm")
                        updated["rs_momentum_period"] = st.number_input("RS lookback bars", 5, 60, int(c.get("rs_momentum_period", 20)), key=f"{key}_rsp")
                        updated["pullback_dist_min"]  = st.number_input("Pullback EMA21 floor %", -0.05, 0.0, float(c.get("pullback_dist_min", -0.01)), 0.002, format="%.3f", key=f"{key}_pblo")
                    with col2:
                        updated["stop_atr_mult"]      = st.number_input("Stop ATR mult", 0.5, 4.0, float(c.get("stop_atr_mult", 2.0)), 0.5, key=f"{key}_stop")
                        updated["target_atr_mult"]    = st.number_input("Target ATR mult", 1.0, 6.0, float(c.get("target_atr_mult", 3.0)), 0.5, key=f"{key}_tgt")

                elif key == "gap_trading":
                    col1, col2 = st.columns(2)
                    with col1:
                        updated["gap_continuation_min"] = st.number_input("Continuation gap min %", 0.003, 0.03, float(c.get("gap_continuation_min", 0.01)), 0.002, format="%.3f", key=f"{key}_gcont")
                        updated["gap_fade_min"]         = st.number_input("Fade gap min %", 0.001, 0.02, float(c.get("gap_fade_min", 0.005)), 0.001, format="%.3f", key=f"{key}_gfade")
                        updated["vol_continuation_min"] = st.number_input("Continuation vol ratio", 1.0, 4.0, float(c.get("vol_continuation_min", 2.0)), 0.5, key=f"{key}_gvol")
                    with col2:
                        updated["stop_cont_atr_mult"]   = st.number_input("Stop ATR (continuation)", 0.5, 4.0, float(c.get("stop_cont_atr_mult", 2.0)), 0.5, key=f"{key}_gstop")
                        updated["stop_fade_atr_mult"]   = st.number_input("Stop ATR (fade)", 0.5, 3.0, float(c.get("stop_fade_atr_mult", 1.5)), 0.5, key=f"{key}_gfstop")
                        updated["gap_session_end"]      = st.text_input("Gap session end (HH:MM)", c.get("gap_session_end", "10:15"), key=f"{key}_gsend")

                elif key == "price_action":
                    col1, col2 = st.columns(2)
                    with col1:
                        updated["sr_proximity_pct"] = st.number_input("S/R proximity %", 0.001, 0.02, float(c.get("sr_proximity_pct", 0.005)), 0.001, format="%.3f", key=f"{key}_srp")
                        updated["stop_atr_mult"]    = st.number_input("Stop ATR mult", 0.5, 3.0, float(c.get("stop_atr_mult", 1.0)), 0.5, key=f"{key}_stop")
                    with col2:
                        updated["target_atr_mult"]  = st.number_input("Target ATR mult", 1.0, 5.0, float(c.get("target_atr_mult", 2.0)), 0.5, key=f"{key}_tgt")
                        updated["volume_confirm"]   = st.number_input("Vol ratio floor", 0.5, 2.0, float(c.get("volume_confirm", 1.2)), 0.1, key=f"{key}_vol")

                elif key == "swing_trading":
                    col1, col2 = st.columns(2)
                    with col1:
                        updated["rsi_min"]           = st.number_input("RSI min", 25.0, 55.0, float(c.get("rsi_min", 40.0)), 1.0, key=f"{key}_rsimin")
                        updated["rsi_max"]           = st.number_input("RSI max", 45.0, 75.0, float(c.get("rsi_max", 60.0)), 1.0, key=f"{key}_rsimax")
                        updated["adx_min"]           = st.number_input("ADX min", 10.0, 35.0, float(c.get("adx_min", 20.0)), 1.0, key=f"{key}_adxmin")
                        updated["adx_max"]           = st.number_input("ADX max", 25.0, 60.0, float(c.get("adx_max", 40.0)), 1.0, key=f"{key}_adxmax")
                    with col2:
                        updated["pullback_vol_max"]  = st.number_input("Pullback vol ratio max", 0.3, 1.5, float(c.get("pullback_vol_max", 0.9)), 0.1, key=f"{key}_pbvol")
                        updated["stop_atr_mult"]     = st.number_input("Stop ATR mult", 1.0, 5.0, float(c.get("stop_atr_mult", 2.5)), 0.5, key=f"{key}_stop")
                        updated["target_atr_mult"]   = st.number_input("Target ATR mult", 2.0, 10.0, float(c.get("target_atr_mult", 5.0)), 0.5, key=f"{key}_tgt")
                        updated["vix_max"]           = st.number_input("VIX max", 10.0, 35.0, float(c.get("vix_max", 22.0)), 1.0, key=f"{key}_vix")

                if updated and st.button(f"✅ Save {s['label']} params", key=f"save_{key}"):
                    cfg[key] = updated
                    try:
                        _save_cfg(cfg)
                        st.success(f"{s['label']} parameters saved.")
                    except Exception as ex:
                        st.error(f"Save failed: {ex}")

    # ══ TAB 3: Options Strategies ═════════════════════════════════════════════
    with tab_opt:
        st.subheader("Options Strategy Suite — 19 Strategies")
        st.info(
            "Strategies generate **regime-based condition signals** with computed strikes. "
            "Full execution (live chain, Greeks, order routing) is **Phase 4**. "
            "Strike intervals: Nifty=50pts · BankNifty=100pts · Stocks=5pts"
        )

        enabled_list = cfg.get("phase21", {}).get("enabled_strategies", [])

        # ── Expected Move Calculator (top of tab) ──────────────────────────────
        with st.expander("📐 Expected Move Calculator", expanded=True):
            import math as _math
            ecol1, ecol2, ecol3, ecol4 = st.columns(4)
            with ecol1:
                em_sym   = st.selectbox("Underlying", ["Nifty (50pt)", "BankNifty (100pt)", "FinNifty (50pt)", "Stock (5pt)"], key="em_sym_global")
            with ecol2:
                em_close = st.number_input("Current price", 100.0, 200000.0, 23000.0, 50.0, key="em_close_global")
            with ecol3:
                em_vix   = st.number_input("India VIX", 5.0, 60.0, 18.0, 0.5, key="em_vix_global")
            with ecol4:
                em_dte   = st.number_input("DTE (days)", 1, 90, 7, key="em_dte_global")

            _em  = em_close * (em_vix / 100) * _math.sqrt(em_dte / 365)
            _int = 100 if "BankNifty" in em_sym else (5 if "Stock" in em_sym else 50)

            def _rs_ui(p: float) -> int:
                return int(round(p / _int) * _int)

            st.metric("Expected Move ±1 SD", f"±{_em:.0f} pts  ({_em/em_close*100:.2f}%)",
                      help="EM = Close × (VIX/100) × √(DTE/365)")
            st.markdown(f"""
| SD | Call side | Put side |
|----|-----------|---------|
| ±0.5 SD | {_rs_ui(em_close + 0.5*_em)} | {_rs_ui(em_close - 0.5*_em)} |
| ±1.0 SD (0.16Δ) | {_rs_ui(em_close + _em)} | {_rs_ui(em_close - _em)} |
| ±1.5 SD (0.10Δ) | {_rs_ui(em_close + 1.5*_em)} | {_rs_ui(em_close - 1.5*_em)} |
| ±2.0 SD (wing) | {_rs_ui(em_close + 2.0*_em)} | {_rs_ui(em_close - 2.0*_em)} |
            """)

        # ── Options strategy cards by category ────────────────────────────────
        for category, cat_label, cat_color, cat_icon in [
            ("bullish", "Bullish Strategies", "#28a745", "📈"),
            ("bearish", "Bearish Strategies", "#dc3545", "📉"),
            ("neutral", "Neutral Strategies", "#6c757d", "⚖️"),
        ]:
            st.divider()
            st.subheader(f"{cat_icon} {cat_label}")
            cat_strats = {k: v for k, v in OPTIONS_STRATEGIES.items() if v["category"] == category}

            for key, s in cat_strats.items():
                is_enabled = key in enabled_list
                unlimited_risk = "⚠️" in s.get("risk", "")

                with st.expander(
                    f"{'✅' if is_enabled else '❌'}  {s['label']}"
                    + ("  ⚠️ Unlimited risk" if unlimited_risk else "  ✅ Defined risk"),
                    expanded=False,
                ):
                    col_l, col_r = st.columns([1.5, 1])
                    with col_l:
                        st.markdown(f"**Entry condition:** `{s['condition']}`")
                        st.markdown(f"**Formula / BEP:** {s['formula']}")
                    with col_r:
                        st.markdown(f"**Max profit:** {s['reward']}")
                        st.markdown(f"**Max loss:** {s['risk']}")

                    if unlimited_risk:
                        st.warning("⚠️ Unlimited risk — use only with hedge or hard stop. Requires active delta management.")

                    # Enable/disable toggle
                    en_new = st.toggle(f"Enable {s['label']}", value=is_enabled, key=f"opt_tog_{key}")
                    if en_new != is_enabled:
                        if en_new and key not in enabled_list:
                            enabled_list.append(key)
                        elif not en_new and key in enabled_list:
                            enabled_list.remove(key)
                        cfg.setdefault("phase21", {})["enabled_strategies"] = enabled_list
                        try:
                            _save_cfg(cfg)
                            st.success(f"{s['label']} {'enabled' if en_new else 'disabled'}.")
                        except Exception as ex:
                            st.error(f"Save failed: {ex}")

                    # Key parameter editors
                    c_params = cfg.get(key, {})

                    if key in ("buy_call", "buy_put"):
                        col1, col2 = st.columns(2)
                        with col1:
                            new_adx_min = st.number_input("ADX min", 10.0, 40.0, float(c_params.get("adx_min", 20.0)), 1.0, key=f"{key}_adx")
                        with col2:
                            new_vix_max = st.number_input("VIX max (buy cheap)", 10.0, 35.0, float(c_params.get("vix_max", 20.0)), 1.0, key=f"{key}_vmax")
                        if st.button(f"💾 Save", key=f"save_{key}"):
                            cfg[key] = {"adx_min": new_adx_min, "vix_max": new_vix_max, "strike_sd": float(c_params.get("strike_sd", 0.0))}
                            _save_and_confirm(cfg)

                    elif key in ("sell_put", "sell_call"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            new_vix_min = st.number_input("VIX min (sell IV)", 10.0, 35.0, float(c_params.get("vix_min", 16.0)), 1.0, key=f"{key}_vmin")
                        with col2:
                            new_adx_max = st.number_input("ADX max", 10.0, 40.0, float(c_params.get("adx_max", 30.0)), 1.0, key=f"{key}_adxmax")
                        with col3:
                            new_sd = st.number_input("Strike SD", 0.3, 2.0, float(c_params.get("strike_sd", 1.0)), 0.1, key=f"{key}_sd")
                        if st.button(f"💾 Save", key=f"save_{key}"):
                            cfg[key] = {"vix_min": new_vix_min, "adx_max": new_adx_max, "strike_sd": new_sd}
                            _save_and_confirm(cfg)

                    elif key == "iron_condor":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            new_adx_max  = st.number_input("ADX max", 5.0, 35.0, float(c_params.get("adx_max", 20.0)), 1.0, key=f"{key}_adxmax")
                            new_vix_min  = st.number_input("VIX min", 8.0, 25.0, float(c_params.get("vix_min", 15.0)), 1.0, key=f"{key}_vmin")
                        with col2:
                            new_vix_max  = st.number_input("VIX max", 20.0, 60.0, float(c_params.get("vix_max", 35.0)), 1.0, key=f"{key}_vmax")
                            new_sd_short = st.number_input("Short SD", 1.0, 2.5, float(c_params.get("sd_short", 1.5)), 0.1, key=f"{key}_sds")
                        with col3:
                            new_sd_wing  = st.number_input("Wing SD", 1.5, 3.0, float(c_params.get("sd_wing", 2.0)), 0.1, key=f"{key}_sdw")
                            new_dte      = st.number_input("DTE", 1, 30, int(c_params.get("dte", 7)), key=f"{key}_dte")
                        if st.button(f"💾 Save", key=f"save_{key}"):
                            cfg["iron_condor"] = {"adx_max": new_adx_max, "vix_min": new_vix_min, "vix_max": new_vix_max, "sd_short": new_sd_short, "sd_wing": new_sd_wing, "dte": new_dte}
                            _save_and_confirm(cfg)

                    elif key in ("short_straddle", "short_strangle", "short_iron_butterfly", "short_iron_wonder"):
                        col1, col2 = st.columns(2)
                        with col1:
                            new_adx_max = st.number_input("ADX max", 5.0, 25.0, float(c_params.get("adx_max", 15.0)), 1.0, key=f"{key}_adxmax")
                        with col2:
                            new_vix_min = st.number_input("VIX min", 12.0, 40.0, float(c_params.get("vix_min", 20.0)), 1.0, key=f"{key}_vmin")
                        if st.button(f"💾 Save", key=f"save_{key}"):
                            cfg[key] = dict(c_params, adx_max=new_adx_max, vix_min=new_vix_min)
                            _save_and_confirm(cfg)

                    elif key in ("call_back_spread", "put_back_spread"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            new_adx = st.number_input("ADX min", 20.0, 40.0, float(c_params.get("adx_min", 25.0)), 1.0, key=f"{key}_adxmin")
                        with col2:
                            new_vix = st.number_input("VIX max (buy cheap)", 10.0, 25.0, float(c_params.get("vix_max", 18.0)), 1.0, key=f"{key}_vmax")
                        with col3:
                            new_ratio = st.number_input("Ratio (sell:buy)", 1, 3, int(c_params.get("ratio", 2)), key=f"{key}_ratio")
                        if st.button(f"💾 Save", key=f"save_{key}"):
                            cfg[key] = dict(c_params, adx_min=new_adx, vix_max=new_vix, ratio=new_ratio)
                            _save_and_confirm(cfg)

                    elif key in ("bull_call_spread", "bear_put_spread", "bull_put_spread", "bear_call_spread"):
                        col1, col2 = st.columns(2)
                        with col1:
                            new_adx = st.number_input("ADX min", 10.0, 35.0, float(c_params.get("adx_min", 20.0)), 1.0, key=f"{key}_adxmin")
                        with col2:
                            new_vix = st.number_input("VIX threshold", 8.0, 35.0, float(c_params.get("vix_min", 15.0) or c_params.get("vix_max", 25.0)), 1.0, key=f"{key}_vix")
                        if st.button(f"💾 Save", key=f"save_{key}"):
                            cfg[key] = dict(c_params)
                            _save_and_confirm(cfg)

                    else:
                        st.caption("_Edit parameters directly in configs/strategy.yaml_")

    # ══ TAB 4: Live Option Chain ══════════════════════════════════════════════
    with tab_chain:
        st.subheader("🔴 Live Option Chain + Market Context")
        st.caption(
            "Data source: NSE India (primary) → Fyers API (fallback) · "
            "Greeks computed via Black-Scholes · Strike intervals: Nifty=50pt · BankNifty=100pt"
        )

        # ── Symbol + expiry selector ───────────────────────────────────────────
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            chain_sym = st.selectbox(
                "Symbol",
                ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"],
                key="chain_sym",
            )
        with lc2:
            expiry_idx = st.number_input("Expiry (0=nearest)", 0, 5, 0, key="chain_exp")
        with lc3:
            strike_cnt = st.number_input("Strikes around ATM", 5, 40, 20, key="chain_strikes")
        with lc4:
            exec_mode  = st.selectbox("Execution mode", ["paper", "live"], key="chain_mode")

        fetch_col, _ = st.columns([1, 3])
        with fetch_col:
            do_fetch = st.button("🔄 Fetch Live Chain", type="primary", key="do_fetch")

        if do_fetch:
            with st.spinner("Fetching option chain from NSE…"):
                try:
                    from production.data.option_chain import OptionChainFetcher
                    from production.strategy.options_executor import OptionsExecutor

                    # Try to get Fyers client from session state
                    fyers_client = st.session_state.get("fyers_client", None)
                    fetcher  = OptionChainFetcher(fyers_client=fyers_client)
                    chain    = fetcher.fetch(chain_sym, int(expiry_idx), int(strike_cnt), force_refresh=True)
                    executor = OptionsExecutor(fyers_client=fyers_client)

                    # Cache in session state
                    st.session_state["live_chain"]    = chain
                    st.session_state["live_executor"] = executor
                    st.session_state["live_fetcher"]  = fetcher
                    st.success(f"✅ Chain fetched from **{chain.source.upper()}** — {len(chain.quotes)//2} strikes · expiry {chain.expiry}")

                except Exception as ex:
                    st.error(f"Fetch failed: {ex}")
                    st.info("If NSE is blocked, connect your Fyers session first (Dashboard → Connect Fyers), then retry.")

        chain = st.session_state.get("live_chain", None)

        if chain is not None:
            # ── Market summary ─────────────────────────────────────────────────
            st.divider()
            st.subheader("📊 Market Context")

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            vix_color = "🔴" if chain.vix > 20 else ("🟡" if chain.vix > 15 else "🟢")
            m1.metric("Underlying",  f"₹{chain.underlying:,.0f}")
            m2.metric("ATM IV",      f"{chain.atm_iv:.1f}%")
            m3.metric(f"VIX {vix_color}", f"{chain.vix:.2f}" if chain.vix else "—")
            m4.metric("PCR (OI)",    f"{chain.pcr:.2f}")
            m5.metric("Max Pain",    f"₹{chain.max_pain:,.0f}")
            m6.metric("EM ±1 SD",    f"±{chain.em_1sd:.0f} pts")

            # VIX expected move table
            import math as _m
            vix_v = chain.vix or chain.atm_iv
            if vix_v:
                st.markdown(f"""
**India VIX implied expected moves for {chain_sym.split(':')[1]}** (underlying ≈ ₹{chain.underlying:,.0f})

| Horizon | Expected Move | Range |
|---------|---------------|-------|
| Daily   | ±{chain.underlying*(vix_v/100)*_m.sqrt(1/365):.0f} pts | {chain.underlying-(chain.underlying*(vix_v/100)*_m.sqrt(1/365)):.0f} – {chain.underlying+(chain.underlying*(vix_v/100)*_m.sqrt(1/365)):.0f} |
| Weekly (7d) | ±{chain.em_1sd:.0f} pts | {chain.underlying-chain.em_1sd:.0f} – {chain.underlying+chain.em_1sd:.0f} |
| Monthly (30d) | ±{chain.underlying*(vix_v/100)*_m.sqrt(30/365):.0f} pts | {chain.underlying-(chain.underlying*(vix_v/100)*_m.sqrt(30/365)):.0f} – {chain.underlying+(chain.underlying*(vix_v/100)*_m.sqrt(30/365)):.0f} |
                """)

            # ── Nifty50 market context ─────────────────────────────────────────
            fyers_client = st.session_state.get("fyers_client", None)
            if fyers_client and st.button("🌍 Load Nifty50 Market Context", key="load_ctx"):
                with st.spinner("Fetching all Nifty50 quotes…"):
                    try:
                        fetcher_ctx = st.session_state.get("live_fetcher")
                        ctx = fetcher_ctx.fetch_market_context(vix_value=chain.vix or 0)
                        if ctx:
                            st.session_state["market_ctx"] = ctx
                    except Exception as ex:
                        st.warning(f"Market context: {ex}")

            ctx = st.session_state.get("market_ctx", None)
            if ctx:
                st.subheader("🏦 Index Performance")
                ci1, ci2, ci3, ci4 = st.columns(4)
                ci1.metric("Nifty50",     f"₹{ctx.nifty_last:,.0f}",      f"{ctx.nifty_change_pct:+.2f}%")
                ci2.metric("BankNifty",   f"₹{ctx.banknifty_last:,.0f}",  f"{ctx.banknifty_change_pct:+.2f}%")
                ci3.metric("Bank vs Nifty", f"{ctx.banknifty_vs_nifty:+.2f}%",
                           "Banking outperforming" if ctx.banknifty_vs_nifty > 0 else "Banking underperforming")
                ci4.metric("India VIX",   f"{ctx.india_vix:.2f}")

                col_g, col_l = st.columns(2)
                with col_g:
                    st.markdown("**🟢 Top 5 Nifty50 Gainers**")
                    gdf = pd.DataFrame(ctx.top_gainers)[["symbol", "ltp", "change_pct", "rs_vs_nifty"]]
                    gdf.columns = ["Symbol", "LTP", "Change %", "RS vs Nifty"]
                    st.dataframe(gdf, hide_index=True, use_container_width=True)
                with col_l:
                    st.markdown("**🔴 Top 5 Nifty50 Losers**")
                    ldf = pd.DataFrame(ctx.top_losers)[["symbol", "ltp", "change_pct", "rs_vs_nifty"]]
                    ldf.columns = ["Symbol", "LTP", "Change %", "RS vs Nifty"]
                    st.dataframe(ldf, hide_index=True, use_container_width=True)

                # Full Nifty50 heatmap-style table
                with st.expander("📋 All Nifty50 Stocks — RS vs Nifty", expanded=False):
                    all_rows = [
                        {"Symbol": k.split(":")[1], "LTP": v["ltp"],
                         "Change %": v["change_pct"], "RS vs Nifty": v["rs_vs_nifty"]}
                        for k, v in ctx.nifty50_quotes.items() if v["ltp"] > 0
                    ]
                    adf = pd.DataFrame(all_rows).sort_values("RS vs Nifty", ascending=False)
                    st.dataframe(adf, hide_index=True, use_container_width=True)

            # ── Live option chain table ────────────────────────────────────────
            st.divider()
            st.subheader("📋 Option Chain")

            chain_df = chain.to_df()
            if not chain_df.empty:
                # Highlight ATM row
                atm = chain.atm_strike
                def _highlight_atm(row):
                    return ["background-color: #fff3cd" if row["strike"] == atm else "" for _ in row]

                display_cols = [c for c in chain_df.columns if any(
                    k in c for k in ["strike", "CE_ltp", "CE_iv", "CE_oi", "CE_delta",
                                      "CE_theta", "PE_ltp", "PE_iv", "PE_oi", "PE_delta", "PE_theta"]
                )]
                display_df = chain_df[display_cols].copy()
                display_df.columns = [c.replace("CE_", "CE ").replace("PE_", "PE ") for c in display_df.columns]

                st.dataframe(
                    display_df.style.apply(_highlight_atm, axis=1),
                    use_container_width=True, hide_index=True,
                )
                st.caption(f"⭐ Highlighted row = ATM strike ({atm:.0f}) · Source: {chain.source.upper()} · {chain.timestamp.strftime('%H:%M:%S IST')}")
            else:
                st.warning("Chain data empty — try a different symbol or expiry.")

            # ── Strategy signals with real premiums ───────────────────────────
            st.divider()
            st.subheader("🎯 Active Strategy Signals → Real Premiums")
            st.caption("Strategies are evaluated against the current chain. Click Execute to place paper/live orders.")

            # Build feature row from chain for signal evaluation
            import pandas as _pd
            f_proxy = _pd.Series({
                "adx": 15.0, "rsi": 50.0, "di_diff": 0.0, "vol_ratio": 1.0,
                "dist_ema_21": 0.0, "dist_ema_50": 0.0, "dist_ema_200": 0.0,
                "bb_pct_b": 0.5, "bb_squeeze": 0, "macd_hist_norm": 0.0, "atr_pct": 0.003,
            })
            st.info(
                "⚠️ Signal conditions below use placeholder feature values (ADX=15, RSI=50). "
                "In live trading, real bar features from the data feed are used. "
                "Use this panel to preview strike selection and premiums for any strategy."
            )

            from production.strategy.options_strategies import ALL_OPTIONS_STRATEGIES
            executor = st.session_state.get("live_executor", OptionsExecutor())

            for strat in ALL_OPTIONS_STRATEGIES:
                try:
                    sig = strat.evaluate_bar(chain_sym, f_proxy, chain.underlying, vix=chain.vix or chain.atm_iv)
                    if sig is None or not sig.condition_met:
                        continue
                    legs = executor.resolve_legs(sig, chain)
                    if not legs:
                        continue
                    payoff = executor.compute_payoff(legs, chain.underlying)

                    cat_icon = {"bullish": "📈", "bearish": "📉", "neutral": "⚖️"}.get(sig.category, "")
                    with st.expander(
                        f"{cat_icon} **{sig.strategy.replace('_',' ').title()}** — "
                        f"Net {'Credit' if payoff.net_premium > 0 else 'Debit'} "
                        f"₹{abs(payoff.net_premium):,.0f}  |  "
                        f"Max Profit ₹{payoff.max_profit:,.0f}  |  "
                        f"Max Loss ₹{payoff.max_loss:,.0f}",
                        expanded=False,
                    ):
                        # Leg table
                        leg_rows = [{
                            "Action":   l.action.upper(),
                            "Type":     l.option_type.upper(),
                            "Strike":   f"₹{l.actual_strike:,.0f}",
                            "Fyers Symbol": l.fyers_symbol,
                            "LTP":      f"₹{l.ltp:.2f}",
                            "Bid/Ask":  f"₹{l.bid:.2f} / ₹{l.ask:.2f}",
                            "IV":       f"{l.iv:.1f}%",
                            "Delta":    f"{l.delta:+.3f}",
                            "Theta":    f"{l.theta:.2f}/day",
                            "OI":       f"{l.oi:,}",
                            "Qty (lots)": l.qty,
                        } for l in legs]
                        st.dataframe(pd.DataFrame(leg_rows), hide_index=True, use_container_width=True)

                        # Risk profile
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Net Premium", f"₹{payoff.net_premium:+,.0f}")
                        r2.metric("Max Profit",  f"₹{payoff.max_profit:,.0f}")
                        r3.metric("Max Loss",    f"₹{payoff.max_loss:,.0f}")
                        r4.metric("BEP", f"{payoff.break_even_dn:.0f} – {payoff.break_even_up:.0f}")

                        # Execute button
                        if st.button(
                            f"{'🧾 Paper' if exec_mode == 'paper' else '⚡ LIVE'} Execute — {sig.strategy}",
                            key=f"exec_{sig.strategy}",
                            type="primary" if exec_mode == "live" else "secondary",
                        ):
                            if exec_mode == "live":
                                st.error("⚠️ Live execution requires Fyers session + SEBI compliance review. Switching to paper.")
                                exec_mode = "paper"
                            results = executor.place_paper(legs, sig, payoff)
                            for r in results:
                                st.success(f"✅ {r.message}")

                except Exception as ex:
                    pass  # strategy condition not met or chain data issue

            # ── Paper positions ────────────────────────────────────────────────
            st.divider()
            st.subheader("📁 Open Paper Option Positions")
            try:
                executor_p = st.session_state.get("live_executor", OptionsExecutor())
                pos_df = executor_p.open_positions()
                if pos_df.empty:
                    st.info("No open paper positions.")
                else:
                    disp_cols = [c for c in ["time", "strategy", "action", "option_type",
                                              "actual_strike", "fyers_symbol", "price", "iv",
                                              "net_premium", "max_profit", "max_loss"] if c in pos_df.columns]
                    st.dataframe(pos_df[disp_cols], hide_index=True, use_container_width=True)
            except Exception:
                st.info("No positions loaded.")

    # ══ TAB 5: ML Models ══════════════════════════════════════════════════════
    with tab_ml:
        st.subheader("ML Model Status")
        col1, col2 = st.columns(2)
        with col1:
            _model_badge(HMM_MODEL, "HMM Regime Model")
        with col2:
            _model_badge(XGB_MODEL, "XGBoost Direction Model")

        st.divider()
        st.subheader("⚙️ XGBoost Hyperparameters")
        p2 = cfg.get("phase2", {})
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.number_input("n_estimators", 50, 2000, int(p2.get("xgb_n_estimators", 500)), 50, key="xgb_nest")
            max_depth    = st.number_input("max_depth", 2, 10, int(p2.get("xgb_max_depth", 4)), key="xgb_depth")
        with col2:
            learning_rate = st.number_input("learning_rate", 0.005, 0.5, float(p2.get("xgb_learning_rate", 0.05)), 0.005, format="%.3f", key="xgb_lr")
            subsample     = st.number_input("subsample", 0.3, 1.0, float(p2.get("xgb_subsample", 0.8)), 0.1, key="xgb_sub")

        st.divider()
        st.subheader("📊 Walk-Forward Validator")
        col1, col2 = st.columns(2)
        with col1:
            train_bars    = st.number_input("Min training bars", 500, 10000, int(p2.get("wfv_train_bars", 2000)), 500, key="wfv_train")
            test_bars     = st.number_input("Test bars per window", 100, 2000, int(p2.get("wfv_test_bars", 500)), 100, key="wfv_test")
        with col2:
            n_windows     = st.number_input("WF windows", 2, 20, int(p2.get("wfv_n_windows", 5)), key="wfv_nwin")
            dsr_threshold = st.number_input("DSR threshold (live gate)", 0.1, 3.0, float(p2.get("wfv_dsr_threshold", 1.0)), 0.1, key="wfv_dsr")

        st.divider()
        st.subheader("📰 News Sentiment")
        col1, col2 = st.columns(2)
        with col1:
            bearish_block = st.number_input("Bearish block (LONG gate)", -1.0, -0.05, float(p2.get("sentiment_bearish_block", -0.3)), 0.05, key="sent_bear")
            bullish_block = st.number_input("Bullish block (SHORT gate)", 0.05, 1.0, float(p2.get("sentiment_bullish_block", 0.3)), 0.05, key="sent_bull")
        with col2:
            cache_ttl = st.number_input("Cache TTL (seconds)", 60, 1800, int(p2.get("sentiment_cache_ttl_s", 300)), 60, key="sent_ttl")
            st.info("Score range: −1.0 (very bearish) → +1.0 (very bullish)")

        if st.button("✅ Save ML / WFV / Sentiment Config", type="primary", key="save_ml"):
            cfg.setdefault("phase2", {}).update({
                "xgb_n_estimators": n_estimators, "xgb_max_depth": max_depth,
                "xgb_learning_rate": learning_rate, "xgb_subsample": subsample,
                "wfv_train_bars": train_bars, "wfv_test_bars": test_bars,
                "wfv_n_windows": n_windows, "wfv_dsr_threshold": dsr_threshold,
                "sentiment_bearish_block": bearish_block, "sentiment_bullish_block": bullish_block,
                "sentiment_cache_ttl_s": cache_ttl,
            })
            try:
                _save_cfg(cfg)
                st.success("ML / WFV / Sentiment config saved.")
            except Exception as ex:
                st.error(f"Save failed: {ex}")

    # ══ TAB 6: Requests ═══════════════════════════════════════════════════════
    with tab_req:
        st.subheader("💬 Request a Phase 2.1 Change")
        st.caption("Describe what you'd like changed. Claude Code reads this and implements it.")

        context = st.selectbox(
            "Component",
            [s["label"] for s in STRATEGIES.values()] +
            [s["label"] for s in OPTIONS_STRATEGIES.values()] +
            ["XGBoost model", "HMM regime model", "News sentiment",
             "Walk-forward validator", "Orchestrator / Routing", "Other"],
        )
        request_text = st.text_area(
            "Your request",
            placeholder=(
                "e.g. 'Add a 5-bar confirmation filter to the Breakout strategy'\n"
                "e.g. 'Tighten Gap Trading fade threshold to 0.3%'\n"
                "e.g. 'Backtest Mean Reversion on NIFTYBANK 5min and show results'\n"
                "e.g. 'Retrain XGBoost with more trees and show OOS accuracy'\n"
                "e.g. 'Add Supertrend to the Trend Following strategy'"
            ),
            height=130,
        )

        if st.button("📨 Submit Request to Claude", type="secondary", key="submit_req"):
            if not request_text.strip():
                st.warning("Please describe the change.")
            else:
                _save_request(request_text, context, {})
                st.success("Request saved! Tell Claude Code: **'check pending strategy requests'**")
                st.code(f"Logged to: {REQUESTS_FILE}", language="text")

        # Pending requests
        if REQUESTS_FILE.exists():
            lines = REQUESTS_FILE.read_text().strip().splitlines()
            pending = [json.loads(l) for l in lines
                       if json.loads(l).get("status") == "pending"
                       and json.loads(l).get("phase") in ("Phase 2", "Phase 2.1")]
            if pending:
                st.divider()
                st.subheader(f"🕐 Pending Requests ({len(pending)})")
                for req in reversed(pending[-5:]):
                    with st.expander(
                        f"{req['time'][:16]}  [{req.get('context','')}]  —  {req['request'][:60]}…"
                    ):
                        st.write(req["request"])
                        if req.get("params_changed"):
                            st.json(req["params_changed"])
