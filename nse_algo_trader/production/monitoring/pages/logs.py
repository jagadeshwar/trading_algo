"""Logs page — view system logs, trade logs, and circuit breaker events."""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd

LOG_DIR = Path("logs")
LEVELS = ["ALL", "INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"]
LEVEL_COLORS = {
    "CRITICAL": "#ff4c6a", "ERROR": "#ff4c6a",
    "WARNING": "#ffd166", "SUCCESS": "#00c896",
    "INFO": "#8892a4", "DEBUG": "#555",
}


def _read_log(path: Path, max_lines: int = 500) -> list[dict]:
    lines = []
    try:
        text = path.read_text(errors="replace")
        for line in text.splitlines()[-max_lines:]:
            parts = line.split("|")
            if len(parts) >= 3:
                try:
                    time_str = parts[0].strip()
                    level    = parts[1].strip()
                    msg      = "|".join(parts[2:]).strip()
                    lines.append({"time": time_str, "level": level, "message": msg})
                except Exception:
                    lines.append({"time": "", "level": "INFO", "message": line})
            elif line.strip():
                lines.append({"time": "", "level": "INFO", "message": line})
    except Exception:
        pass
    return lines


def render():
    st.title("📋 Logs")

    # ── File selector ─────────────────────────────────────────────────────────
    if not LOG_DIR.exists() or not list(LOG_DIR.glob("*.log")):
        st.info("No log files found yet. Logs appear here once trading starts.")
        return

    log_files = sorted(LOG_DIR.glob("*.log"), reverse=True)
    file_names = [f.name for f in log_files]

    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        selected = st.selectbox("Log file", file_names)
    with c2:
        level_filter = st.selectbox("Level filter", LEVELS)
    with c3:
        max_lines = st.selectbox("Max lines", [100, 250, 500, 1000], index=1)

    search = st.text_input("🔍 Search logs", placeholder="e.g. CIRCUIT, ENTRY, ERROR...")

    # ── Load and filter ───────────────────────────────────────────────────────
    selected_path = LOG_DIR / selected
    all_lines = _read_log(selected_path, max_lines)

    if level_filter != "ALL":
        all_lines = [l for l in all_lines if l["level"] == level_filter]
    if search:
        all_lines = [l for l in all_lines if search.upper() in l["message"].upper()]

    # ── Stats strip ───────────────────────────────────────────────────────────
    counts = {}
    for l in all_lines:
        counts[l["level"]] = counts.get(l["level"], 0) + 1

    s_cols = st.columns(len(counts) if counts else 1)
    for col, (level, count) in zip(s_cols, counts.items()):
        col.metric(level, count)

    st.divider()

    # ── Render log lines ──────────────────────────────────────────────────────
    if not all_lines:
        st.info("No matching log entries.")
        return

    # Trade-specific highlights
    trade_lines = [l for l in all_lines if "PAPER ENTRY" in l["message"] or "PAPER EXIT" in l["message"]]
    if trade_lines:
        with st.expander(f"📈 Trade events ({len(trade_lines)} entries)"):
            for l in reversed(trade_lines):
                icon = "🟢" if "ENTRY" in l["message"] else "🔴" if "EXIT" in l["message"] else "⚪"
                pnl_color = ""
                if "PnL=" in l["message"]:
                    try:
                        pnl_str = [p for p in l["message"].split() if p.startswith("PnL=")][0]
                        pnl_val = float(pnl_str.replace("PnL=", "").replace(",", "").replace("₹", ""))
                        pnl_color = "color: #00c896" if pnl_val > 0 else "color: #ff4c6a"
                    except Exception:
                        pass
                st.markdown(f"{icon} `{l['time'][:19]}` — {l['message']}", unsafe_allow_html=False)

    st.divider()

    # Full log as dataframe
    df = pd.DataFrame(reversed(all_lines))
    if df.empty:
        return

    # Colour-code by level — rename first, then style using renamed column "Level"
    renamed = df[["time", "level", "message"]].rename(
        columns={"time": "Time", "level": "Level", "message": "Message"}
    )

    def style_row(row):
        color = LEVEL_COLORS.get(row["Level"], "#8892a4")
        return [f"color: {color}"] * len(row)

    st.dataframe(
        renamed.style.apply(style_row, axis=1),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    # ── Download button ────────────────────────────────────────────────────────
    st.download_button(
        "⬇️  Download log file",
        data=selected_path.read_bytes(),
        file_name=selected,
        mime="text/plain",
    )
