"""Phase 2 Strategy page — HMM regime, XGBoost ML signal, news sentiment, walk-forward validator."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import yaml

CONFIG_PATH   = Path("configs/strategy.yaml")
REQUESTS_FILE = Path("data/strategy_requests.jsonl")
XGB_MODEL     = Path("models/xgb_direction.pkl")
HMM_MODEL     = Path("models/regime_hmm.pkl")
IST           = ZoneInfo("Asia/Kolkata")


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
        "phase":          "Phase 2",
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
        st.warning(f"⚠️ {label} — **not trained yet**. Run training to enable.")


def render() -> None:
    st.title("🤖 Phase 2 — ML-Enhanced Strategy")
    st.caption("HMM regime detection · XGBoost direction model · News sentiment · Walk-forward validation")

    cfg = _load_cfg()
    mr  = cfg.get("mean_reversion", {})
    p2  = cfg.get("phase2", {})

    # ── Overview ──────────────────────────────────────────────────────────────
    with st.expander("📖 Phase 2 Overview", expanded=True):
        st.markdown("""
Phase 2 layers **machine learning** on top of the Phase 1 EMA crossover to
improve signal quality and reduce false entries.

| Component | What it does | Live status |
|-----------|-------------|-------------|
| **HMM Regime Detector** | Classifies market into RANGING / TRENDING / HIGH_VOL using Gaussian HMM trained on returns, ATR, ADX, and volume | Active — labels every bar |
| **XGBoost Classifier** | 3-bar-ahead direction model (Up / Flat / Down) trained with PurgedKFold. Boosts confidence on agreement, penalises on conflict | Active when model file exists |
| **News Sentiment** | Scores NSE announcements + Moneycontrol + ET headlines via VADER. Blocks trades when sentiment strongly opposes direction | Active every bar |
| **Walk-Forward Validator** | Expanding-window OOS backtest with Deflated Sharpe Ratio (DSR) gate. Strategy must pass DSR > 1.0 before going live | Run manually |

**How XGBoost integrates with Phase 1:**
```
Phase 1 signal fires → XGBoost agrees → confidence += 15%
Phase 1 signal fires → XGBoost disagrees → confidence −= 10%
confidence < threshold → trade blocked
```
        """)

    # ── Model status ──────────────────────────────────────────────────────────
    st.subheader("🔬 Model Status")
    c1, c2 = st.columns(2)
    with c1:
        _model_badge(HMM_MODEL,  "HMM Regime Model")
    with c2:
        _model_badge(XGB_MODEL,  "XGBoost Direction Model")

    st.divider()

    # ── Mean Reversion params (Phase 2 alternate strategy) ────────────────────
    st.subheader("⚙️ Mean Reversion Parameters")
    st.caption("Bollinger Band + RSI reversion strategy — activated during RANGING regimes.")

    col1, col2 = st.columns(2)
    with col1:
        bb_period = st.number_input(
            "Bollinger Band period", min_value=5, max_value=50, step=1,
            value=int(mr.get("bb_period", 20)),
            help="Lookback for BB mean and standard deviation",
        )
        bb_std = st.number_input(
            "BB standard deviations", min_value=1.0, max_value=4.0, step=0.5,
            value=float(mr.get("bb_std", 2.0)),
            help="Width of the bands. 2.0 = ~95% of price within bands",
        )
    with col2:
        rsi_oversold = st.number_input(
            "RSI oversold threshold", min_value=10, max_value=40, step=1,
            value=int(mr.get("rsi_oversold", 30)),
            help="Enter long reversion when RSI drops below this",
        )
        rsi_overbought = st.number_input(
            "RSI overbought threshold", min_value=60, max_value=90, step=1,
            value=int(mr.get("rsi_overbought", 70)),
            help="Enter short reversion when RSI exceeds this",
        )

    if st.button("✅ Apply Mean Reversion Changes", type="primary", key="apply_mr"):
        cfg.setdefault("mean_reversion", {}).update({
            "bb_period": bb_period, "bb_std": bb_std,
            "rsi_oversold": rsi_oversold, "rsi_overbought": rsi_overbought,
        })
        try:
            _save_cfg(cfg)
            st.success("Mean reversion parameters saved to `configs/strategy.yaml`.")
        except Exception as ex:
            st.error(f"Save failed: {ex}")

    st.divider()

    # ── News Sentiment ────────────────────────────────────────────────────────
    st.subheader("📰 News Sentiment Filter")
    st.caption("Blocks trades when headlines strongly oppose the signal direction.")

    col1, col2 = st.columns(2)
    with col1:
        bearish_block = st.number_input(
            "Bearish block threshold (LONG gate)",
            min_value=-1.0, max_value=-0.05, step=0.05,
            value=float(p2.get("sentiment_bearish_block", -0.3)),
            help="Block LONG if sentiment score < this. Current: -0.3",
        )
        bullish_block = st.number_input(
            "Bullish block threshold (SHORT gate)",
            min_value=0.05, max_value=1.0, step=0.05,
            value=float(p2.get("sentiment_bullish_block", 0.3)),
            help="Block SHORT if sentiment score > this. Current: 0.3",
        )
    with col2:
        cache_ttl = st.number_input(
            "Sentiment cache TTL (seconds)",
            min_value=60, max_value=1800, step=60,
            value=int(p2.get("sentiment_cache_ttl_s", 300)),
            help="How often to re-fetch headlines. 300 = every 5 minutes",
        )
        st.info("Score range: −1.0 (very bearish) → +1.0 (very bullish)\n"
                "Neutral zone: abs(score) < threshold → allow all signals")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🌲 XGBoost Model Hyperparameters")
    st.caption("Changes here require a model retrain — submit a Claude request below.")

    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.number_input(
            "n_estimators (trees)", min_value=50, max_value=2000, step=50,
            value=int(p2.get("xgb_n_estimators", 500)),
            help="More trees = better fit but slower training",
        )
        max_depth = st.number_input(
            "max_depth", min_value=2, max_value=10, step=1,
            value=int(p2.get("xgb_max_depth", 4)),
            help="Deeper trees capture complex patterns but overfit more",
        )
    with col2:
        learning_rate = st.number_input(
            "learning_rate", min_value=0.005, max_value=0.5, step=0.005,
            value=float(p2.get("xgb_learning_rate", 0.05)),
            format="%.3f",
            help="Lower = needs more trees but generalises better",
        )
        subsample = st.number_input(
            "subsample", min_value=0.3, max_value=1.0, step=0.1,
            value=float(p2.get("xgb_subsample", 0.8)),
            help="Fraction of training rows used per tree. Reduces overfitting",
        )

    # ── Walk-Forward Validator ────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Walk-Forward Validator")
    st.caption("Expanding-window out-of-sample backtest with DSR gate.")

    col1, col2 = st.columns(2)
    with col1:
        train_bars = st.number_input(
            "Min training bars per window", min_value=500, max_value=10000, step=500,
            value=int(p2.get("wfv_train_bars", 2000)),
            help="Minimum bars needed before the first test window",
        )
        test_bars = st.number_input(
            "Test bars per window", min_value=100, max_value=2000, step=100,
            value=int(p2.get("wfv_test_bars", 500)),
            help="How many bars each out-of-sample test window covers",
        )
    with col2:
        n_windows = st.number_input(
            "Number of WF windows", min_value=2, max_value=20, step=1,
            value=int(p2.get("wfv_n_windows", 5)),
            help="More windows = more robust DSR estimate but needs more data",
        )
        dsr_threshold = st.number_input(
            "DSR threshold (live-ready gate)", min_value=0.1, max_value=3.0, step=0.1,
            value=float(p2.get("wfv_dsr_threshold", 1.0)),
            help="Deflated Sharpe Ratio must exceed this to clear for live trading",
        )

    if st.button("✅ Apply Phase 2 Config Changes", type="primary", key="apply_p2"):
        cfg.setdefault("phase2", {}).update({
            "sentiment_bearish_block": bearish_block,
            "sentiment_bullish_block": bullish_block,
            "sentiment_cache_ttl_s":   cache_ttl,
            "xgb_n_estimators":        n_estimators,
            "xgb_max_depth":           max_depth,
            "xgb_learning_rate":       learning_rate,
            "xgb_subsample":           subsample,
            "wfv_train_bars":          train_bars,
            "wfv_test_bars":           test_bars,
            "wfv_n_windows":           n_windows,
            "wfv_dsr_threshold":       dsr_threshold,
        })
        try:
            _save_cfg(cfg)
            st.success("Phase 2 config saved. XGBoost changes require a retrain — submit a request below.")
        except Exception as ex:
            st.error(f"Save failed: {ex}")

    st.divider()

    # ── Claude Request Form ───────────────────────────────────────────────────
    st.subheader("💬 Request a Phase 2 Change")
    st.caption(
        "Describe what you'd like changed. Claude Code reads this file "
        "and implements it — including retraining models if needed."
    )

    context = st.selectbox(
        "Component",
        ["XGBoost model", "HMM regime model", "News sentiment", "Walk-forward validator",
         "Mean reversion strategy", "Other / general Phase 2"],
    )

    request_text = st.text_area(
        "Your request",
        placeholder=(
            "e.g. 'Retrain XGBoost with n_estimators=800 and max_depth=6'\n"
            "e.g. 'Add a 4th HMM state for BREAKOUT regime'\n"
            "e.g. 'Tighten the sentiment block threshold to ±0.2 and check if it reduces whipsaws'\n"
            "e.g. 'Run walk-forward validation on NIFTYBANK 15min and show results'"
        ),
        height=130,
    )

    if st.button("📨 Submit Request to Claude", type="secondary", key="submit_p2"):
        if not request_text.strip():
            st.warning("Please describe the change you'd like made.")
        else:
            changed = {
                "sentiment_bearish_block": bearish_block,
                "sentiment_bullish_block": bullish_block,
                "sentiment_cache_ttl_s":   cache_ttl,
                "xgb_n_estimators":        n_estimators,
                "xgb_max_depth":           max_depth,
                "xgb_learning_rate":       learning_rate,
                "xgb_subsample":           subsample,
                "wfv_train_bars":          train_bars,
                "wfv_test_bars":           test_bars,
                "wfv_n_windows":           n_windows,
                "wfv_dsr_threshold":       dsr_threshold,
            }
            _save_request(request_text, context, changed)
            st.success(
                "Request saved! Tell Claude Code: "
                "**'check pending strategy requests'** to apply it."
            )
            st.code(f"Logged to: {REQUESTS_FILE}", language="text")

    # ── Pending requests ──────────────────────────────────────────────────────
    if REQUESTS_FILE.exists():
        lines = REQUESTS_FILE.read_text().strip().splitlines()
        pending = [json.loads(l) for l in lines
                   if json.loads(l).get("status") == "pending"
                   and json.loads(l).get("phase") == "Phase 2"]
        if pending:
            st.divider()
            st.subheader(f"🕐 Pending Phase 2 Requests ({len(pending)})")
            for req in reversed(pending[-5:]):
                with st.expander(f"{req['time'][:16]}  [{req.get('context','')}]  —  {req['request'][:60]}..."):
                    st.write(req["request"])
                    if req.get("params_changed"):
                        st.json(req["params_changed"])
