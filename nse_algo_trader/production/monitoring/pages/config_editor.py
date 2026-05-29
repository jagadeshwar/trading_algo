"""Config Editor page — edit and validate YAML config files from the UI."""

from __future__ import annotations
from pathlib import Path
import streamlit as st
import yaml

CONFIGS = {
    "risk.yaml":     ("⚠️ Risk Parameters",     "configs/risk.yaml"),
    "strategy.yaml": ("📊 Strategy Parameters",  "configs/strategy.yaml"),
    "broker.yaml":   ("🔌 Broker & Costs",       "configs/broker.yaml"),
    "features.yaml": ("🔬 Feature Engineering",  "configs/features.yaml"),
}

FIELD_HELP = {
    "max_drawdown_pct": "Pause trading when daily drawdown exceeds this % (recommended: 5%)",
    "daily_loss_pct":   "Hard stop — system halts for the day at this daily loss % (recommended: 2%)",
    "max_capital_use":  "Never use more than this % of capital as margin (recommended: 60%)",
    "kelly_fraction":   "Half-Kelly = 0.5. Lower = more conservative position sizing",
    "fast_ema":         "Fast EMA period for crossover (default: 9)",
    "slow_ema":         "Slow EMA period for crossover (default: 21)",
    "volume_filter_ratio": "Volume must be this × 20-bar average to confirm signal (default: 1.2)",
    "stop_atr_multiplier":   "Stop loss distance = this × ATR (default: 2.0)",
    "target_atr_multiplier": "Take profit distance = this × ATR (default: 3.0)",
}


def render():
    st.title("⚙️ Config Editor")
    st.caption("Edit trading parameters. Changes take effect on next trading session start.")

    tab_names = [v[0] for v in CONFIGS.values()]
    tabs = st.tabs(tab_names)

    for tab, (filename, (label, filepath)) in zip(tabs, CONFIGS.items()):
        with tab:
            path = Path(filepath)
            if not path.exists():
                st.warning(f"`{filepath}` not found.")
                continue

            raw = path.read_text()
            try:
                current = yaml.safe_load(raw)
            except Exception:
                current = {}

            # ── Smart editor for risk.yaml ─────────────────────────────────────
            if filename == "risk.yaml":
                _render_risk_editor(current, path)
            elif filename == "strategy.yaml":
                _render_strategy_editor(current, path)
            else:
                _render_yaml_editor(raw, path, label)


def _render_risk_editor(cfg: dict, path: Path):
    st.subheader("Risk Parameters")
    st.info("These parameters control capital protection. Changes only apply after restarting the trading session.")

    with st.form("risk_form"):
        c1, c2 = st.columns(2)
        with c1:
            max_dd = st.number_input(
                "Max Drawdown % (pause trigger)", 1.0, 20.0,
                float(cfg.get("max_drawdown_pct", 5.0)), 0.5,
                help=FIELD_HELP["max_drawdown_pct"])
            daily_loss = st.number_input(
                "Daily Loss % (hard stop)", 0.5, 10.0,
                float(cfg.get("daily_loss_pct", 2.0)), 0.5,
                help=FIELD_HELP["daily_loss_pct"])
            max_capital = st.number_input(
                "Max Capital Use %", 10.0, 100.0,
                float(cfg.get("max_capital_use", 60.0)), 5.0,
                help=FIELD_HELP["max_capital_use"])
            max_position = st.number_input(
                "Max Single Position %", 1.0, 50.0,
                float(cfg.get("max_position_pct", 10.0)), 1.0)

        with c2:
            kelly = st.number_input(
                "Kelly Fraction (0.5 = half-Kelly)", 0.1, 1.0,
                float(cfg.get("kelly_fraction", 0.5)), 0.1,
                help=FIELD_HELP["kelly_fraction"])
            corr_thresh = st.number_input(
                "Correlation Block Threshold", 0.3, 1.0,
                float(cfg.get("correlation_threshold", 0.7)), 0.05)

            st.markdown("**Circuit breakers:**")
            cb = cfg.get("circuit_breakers", {})
            ws_stale = st.number_input("WS stale (sec)", 5, 120, int(cb.get("websocket_stale_sec", 30)), 5)
            vix_spike = st.number_input("VIX spike % trigger", 10.0, 100.0,
                                        float(cb.get("vix_spike_pct", 30.0)), 5.0)
            max_rejections = st.number_input("Max order rejections", 1, 10,
                                             int(cb.get("max_order_rejections", 3)), 1)

        submitted = st.form_submit_button("💾 Save Risk Config", type="primary", use_container_width=True)
        if submitted:
            cfg.update({
                "max_drawdown_pct": max_dd, "daily_loss_pct": daily_loss,
                "max_capital_use": max_capital, "max_position_pct": max_position,
                "kelly_fraction": kelly, "correlation_threshold": corr_thresh,
            })
            cfg.setdefault("circuit_breakers", {}).update({
                "websocket_stale_sec": ws_stale,
                "vix_spike_pct": vix_spike,
                "max_order_rejections": max_rejections,
            })
            path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            st.success("✅ Risk config saved. Restart trading session to apply.")

    # Reference table
    with st.expander("📖 What these mean"):
        st.markdown("""
| Parameter | Effect |
|-----------|--------|
| **Max Drawdown %** | If equity drops X% below today's peak → pause trading for the day |
| **Daily Loss %** | If today's total loss exceeds X% → full stop for the day (hard stop) |
| **Max Capital Use %** | Never have more than X% of capital tied up in margin at once |
| **Max Single Position %** | No single trade uses more than X% of capital |
| **Kelly Fraction** | 0.5 = half-Kelly (safer). 0.25 = quarter-Kelly (very conservative) |
| **Correlation Block** | Don't enter a new trade if it's > X correlated with an open position |
        """)


def _render_strategy_editor(cfg: dict, path: Path):
    st.subheader("Strategy Parameters")

    m = cfg.get("momentum", {})
    r = cfg.get("regime", {})
    v = cfg.get("vix", {})

    with st.form("strategy_form"):
        st.markdown("**EMA Crossover:**")
        sc1, sc2, sc3 = st.columns(3)
        fast_ema = sc1.number_input("Fast EMA", 3, 50, int(m.get("fast_ema", 9)), 1,
                                    help=FIELD_HELP["fast_ema"])
        slow_ema = sc2.number_input("Slow EMA", 5, 200, int(m.get("slow_ema", 21)), 1,
                                    help=FIELD_HELP["slow_ema"])
        vol_ratio= sc3.number_input("Volume filter ratio", 0.5, 5.0,
                                    float(m.get("volume_filter_ratio", 1.2)), 0.1,
                                    help=FIELD_HELP["volume_filter_ratio"])

        st.markdown("**Stop & Target (ATR multiples):**")
        at1, at2 = st.columns(2)
        stop_mult  = at1.number_input("Stop ATR multiplier", 0.5, 10.0,
                                      float(m.get("stop_atr_multiplier", 2.0)), 0.5,
                                      help=FIELD_HELP["stop_atr_multiplier"])
        target_mult= at2.number_input("Target ATR multiplier", 0.5, 10.0,
                                      float(m.get("target_atr_multiplier", 3.0)), 0.5,
                                      help=FIELD_HELP["target_atr_multiplier"])

        rr = target_mult / stop_mult if stop_mult > 0 else 0
        st.caption(f"Reward : Risk ratio = **{rr:.1f}:1** (target should be ≥ 1.5)")

        st.markdown("**Regime filters:**")
        rc1, rc2 = st.columns(2)
        adx_thresh = rc1.number_input("Min ADX (trend strength)", 10, 50,
                                      int(r.get("adx_trending_threshold", 25)), 5)
        vix_max    = rc2.number_input("Max VIX (block above)", 12, 40,
                                      int(v.get("high", 25)), 1)

        submitted = st.form_submit_button("💾 Save Strategy Config", type="primary",
                                           use_container_width=True)
        if submitted:
            cfg.setdefault("momentum", {}).update({
                "fast_ema": fast_ema, "slow_ema": slow_ema,
                "volume_filter_ratio": vol_ratio,
                "stop_atr_multiplier": stop_mult, "target_atr_multiplier": target_mult,
            })
            cfg.setdefault("regime", {})["adx_trending_threshold"] = adx_thresh
            cfg.setdefault("vix", {})["high"] = vix_max
            path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            st.success("✅ Strategy config saved. Restart trading session to apply.")


def _render_yaml_editor(raw: str, path: Path, label: str):
    st.subheader(f"Edit {label}")
    st.caption("Direct YAML editing. Validated before saving.")

    edited = st.text_area("YAML content", raw, height=400, key=str(path))

    col_validate, col_save, _ = st.columns([1, 1, 3])
    with col_validate:
        if st.button("✔️  Validate", key=f"val_{path}"):
            try:
                yaml.safe_load(edited)
                st.success("Valid YAML ✓")
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML: {e}")
    with col_save:
        if st.button("💾 Save", key=f"save_{path}", type="primary"):
            try:
                yaml.safe_load(edited)
                path.write_text(edited)
                st.success(f"Saved to {path}")
            except yaml.YAMLError as e:
                st.error(f"Cannot save — invalid YAML: {e}")

    # Diff view
    if edited != raw:
        with st.expander("📋 Changes preview"):
            import difflib
            diff = list(difflib.unified_diff(
                raw.splitlines(), edited.splitlines(),
                fromfile="original", tofile="edited", lineterm=""
            ))
            if diff:
                st.code("\n".join(diff), language="diff")
