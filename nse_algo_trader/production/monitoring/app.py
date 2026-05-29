"""NSE Algo Trader — Streamlit Dashboard entry point.

Run with:
    streamlit run production/monitoring/app.py
    or
    python run_dashboard.py
"""

import streamlit as st
from pathlib import Path
import sys

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))

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
    st.caption("Phase 1 — Momentum Strategy")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🎮 Trading Controls", "📈 Charts", "🧪 Backtest", "⚙️ Config Editor", "📋 Logs"],
        label_visibility="collapsed",
    )
    st.divider()

    # Quick system status in sidebar
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from production.db.database import ping
        db_ok = ping()
    except Exception:
        db_ok = False

    import json
    from datetime import date
    token_path = Path("fyers_token.txt")
    token_ok = False
    if token_path.exists():
        try:
            data = json.loads(token_path.read_text())
            token_ok = data.get("date") == date.today().isoformat()
        except Exception:
            pass

    st.markdown("**System Status**")
    st.markdown(f"{'🟢' if db_ok else '🔴'} Database {'Connected' if db_ok else 'Disconnected'}")
    st.markdown(f"{'🟢' if token_ok else '🟡'} Fyers Token {'Valid' if token_ok else 'Expired/Missing'}")

    from datetime import datetime
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
    now = datetime.now(IST)
    market_open = now.time() >= __import__('datetime').time(9, 15) and now.time() <= __import__('datetime').time(15, 30) and now.weekday() < 5
    st.markdown(f"{'🟢' if market_open else '🔴'} Market {'Open' if market_open else 'Closed'}")
    st.caption(f"IST: {now.strftime('%H:%M:%S  %d %b %Y')}")

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
elif page == "⚙️ Config Editor":
    from production.monitoring.pages import config_editor
    config_editor.render()
elif page == "📋 Logs":
    from production.monitoring.pages import logs
    logs.render()
