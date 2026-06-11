"""NSE Algo Trader — Streamlit Dashboard entry point.

Run locally:
    streamlit run nse_algo_trader/production/monitoring/app.py   (from repo root)
    streamlit run production/monitoring/app.py                   (from nse_algo_trader/)

Deploy on Streamlit Community Cloud:
    Main file : nse_algo_trader/production/monitoring/app.py
    Branch    : master
    Secrets   : see .streamlit/secrets.toml.example
"""

import os
import streamlit as st
from pathlib import Path
import sys

# ── Path setup ─────────────────────────────────────────────────────────────────
# Make nse_algo_trader/ importable regardless of CWD (local or Streamlit Cloud)
_here = Path(__file__).resolve()
_pkg  = _here.parents[2]   # nse_algo_trader/
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

# ── Streamlit Cloud secrets → env vars ────────────────────────────────────────
# When running on Streamlit Community Cloud, credentials live in st.secrets.
# Promote them to os.environ so existing dotenv-based code works unchanged.
for _k in ("FYERS_APP_ID", "FYERS_SECRET", "FYERS_REDIRECT_URI",
           "FYERS_ACCESS_TOKEN",
           "DATABASE_URL", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "REDIS_URL"):
    try:
        if _k in st.secrets and _k not in os.environ:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass  # key not in secrets or secrets not configured

st.set_page_config(
    page_title="NSE Algo Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 4px 0;
        border-left: 4px solid #00b4d8;
    }
    .metric-label { color: #8892a4; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #e8ecf4; font-size: 24px; font-weight: 700; margin-top: 4px; }
    .metric-value.positive { color: #00c896; }
    .metric-value.negative { color: #ff4c6a; }
    .status-ok    { color: #00c896; font-weight: 700; }
    .status-warn  { color: #ffd166; font-weight: 700; }
    .status-error { color: #ff4c6a; font-weight: 700; }
    .signal-long  { color: #00c896; font-weight: 700; }
    .signal-short { color: #ff4c6a; font-weight: 700; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=64)
    st.title("NSE Algo Trader")
    st.caption("Phase 2 — Multi-Strategy")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "📊 Dashboard",
            "🎮 Trading Controls",
            "📈 Charts",
            "🧪 Backtest",
            "📋 Strategy (Phase 1)",
            "🧠 Strategy Phase 2",
            "⚙️ Config Editor",
            "📋 Logs",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    # Quick system status in sidebar
    import json
    from datetime import date, datetime
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")

    # ── DB status: red only when DATABASE_URL is set but unreachable ──────────
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        # Direct fallback — in case secrets→env promotion ran before st was ready
        try:
            db_url = str(st.secrets.get("DATABASE_URL", "")).strip()
            if db_url:
                os.environ["DATABASE_URL"] = db_url
        except Exception:
            pass
    if not db_url:
        db_status = "⬜ Database not configured"
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from production.db.database import ping
            db_ok = ping()
            db_status = "🟢 Database Connected" if db_ok else "🔴 Database connection failed"
        except Exception:
            db_status = "🔴 Database connection failed"

    # ── Fyers token: env var → session state → fyers_token.txt ───────────────
    token_ok = False
    if os.environ.get("FYERS_ACCESS_TOKEN", "").strip():
        token_ok = True
    elif st.session_state.get("_fyers_token", "").strip():
        # Promote session-state token to env so auth.py picks it up
        os.environ["FYERS_ACCESS_TOKEN"] = st.session_state["_fyers_token"]
        token_ok = True
    else:
        token_path = Path("fyers_token.txt")
        if token_path.exists():
            try:
                data = json.loads(token_path.read_text())
                token_ok = data.get("date") == date.today().isoformat()
            except Exception:
                pass

    now = datetime.now(IST)
    import datetime as _dt
    market_open = (now.time() >= _dt.time(9, 15)
                   and now.time() <= _dt.time(15, 30)
                   and now.weekday() < 5)

    st.markdown("**System Status**")
    st.markdown(db_status)
    st.markdown(f"{'🟢' if token_ok else '🟡'} Fyers Token {'Valid' if token_ok else 'Not set'}")
    st.markdown(f"{'🟢' if market_open else '🔴'} Market {'Open' if market_open else 'Closed'}")
    st.caption(f"IST: {now.strftime('%H:%M:%S  %d %b %Y')}")

    # ── Token paste widget (shown when token is missing) ──────────────────────
    st.divider()
    lbl = "🔑 Update Token" if token_ok else "🔑 Set Fyers Token"
    with st.expander(lbl, expanded=not token_ok):
        st.caption(
            "Run `python nse_algo_trader/auth.py` locally, then paste "
            "the printed token here. Valid until midnight IST."
        )
        pasted = st.text_input("Access token", type="password",
                               key="_token_input_field",
                               placeholder="eyJ0eXAiOiJKV1QiLC...")
        if st.button("Activate token", type="primary"):
            t = pasted.strip()
            if t:
                st.session_state["_fyers_token"] = t
                os.environ["FYERS_ACCESS_TOKEN"] = t
                # Also write to fyers_token.txt so the local cache is warm
                try:
                    Path("fyers_token.txt").write_text(
                        json.dumps({"date": date.today().isoformat(),
                                    "access_token": t})
                    )
                except Exception:
                    pass
                st.success("Token activated ✓")
                st.rerun()
            else:
                st.error("Please paste a token first.")

# ── Page routing ───────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    from production.monitoring.pages import dashboard
    dashboard.render()
elif page == "🎮 Trading Controls":
    from production.monitoring.pages import trading_controls
    trading_controls.render()
elif page == "📈 Charts":
    from production.monitoring.pages import charts
    charts.render()
elif page == "🧪 Backtest":
    from production.monitoring.pages import backtest
    backtest.render()
elif page == "📋 Strategy (Phase 1)":
    from production.monitoring.pages import strategy
    strategy.render()
elif page == "🧠 Strategy Phase 2":
    from production.monitoring.pages import strategy_phase2
    strategy_phase2.render()
elif page == "⚙️ Config Editor":
    from production.monitoring.pages import config_editor
    config_editor.render()
elif page == "📋 Logs":
    from production.monitoring.pages import logs
    logs.render()
