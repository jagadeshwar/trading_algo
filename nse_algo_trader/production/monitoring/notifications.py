"""Notification manager — Telegram, macOS desktop, and Streamlit toast.

Supports three channels (all optional, configurable independently):
  1. Telegram  — mobile push via Bot API
  2. Desktop   — macOS notification centre (osascript)
  3. State file — picked up by Streamlit dashboard as st.toast()

Events notified:
  ENTRY       — signal fired, position opened
  EXIT_WIN    — position closed in profit
  EXIT_LOSS   — position closed at a loss
  CIRCUIT     — circuit breaker tripped
  DAILY       — end-of-day summary
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger

IST          = ZoneInfo("Asia/Kolkata")
STATE_FILE   = Path("data/paper_trader_state.json")
NOTIF_FILE   = Path("data/.latest_notification.json")


# ── Notification payload ───────────────────────────────────────────────────────

@dataclass
class Notification:
    event:   str       # ENTRY | EXIT_WIN | EXIT_LOSS | CIRCUIT | DAILY
    title:   str
    body:    str
    emoji:   str
    time:    str = ""

    def __post_init__(self):
        self.time = datetime.now(IST).strftime("%H:%M:%S")

    @property
    def full_text(self) -> str:
        return f"{self.emoji} {self.title}\n{self.body}"


# ── Manager ────────────────────────────────────────────────────────────────────

class NotificationManager:
    """Send trade notifications across all configured channels."""

    def __init__(self) -> None:
        self.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat  = os.environ.get("TELEGRAM_CHAT_ID",   "")
        self.desktop_enabled = True   # macOS only, silently skipped on other OS

    # ── Public events ─────────────────────────────────────────────────────────

    def notify_entry(
        self,
        symbol: str,
        direction: str,
        price: float,
        stop: float,
        target: float,
        qty: int,
        confidence: float,
        regime: str = "",
    ) -> None:
        rr = abs(target - price) / abs(price - stop) if abs(price - stop) > 0 else 0
        n = Notification(
            event="ENTRY",
            emoji="🟢" if direction == "LONG" else "🔴",
            title=f"PAPER {direction} — {symbol.replace('NSE:','').replace('-EQ','').replace('-INDEX','')}",
            body=(
                f"Entry  : ₹{price:,.2f}  (qty {qty})\n"
                f"Stop   : ₹{stop:,.2f}  (risk ₹{abs(price-stop)*qty:,.0f})\n"
                f"Target : ₹{target:,.2f}  (R:R {rr:.1f})\n"
                f"Conf   : {confidence:.0%}  |  Regime: {regime}"
            ),
        )
        self._dispatch(n)

    def notify_exit(
        self,
        symbol: str,
        price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
        qty: int,
    ) -> None:
        win = pnl > 0
        n = Notification(
            event="EXIT_WIN" if win else "EXIT_LOSS",
            emoji="✅" if win else "❌",
            title=f"{'PROFIT' if win else 'LOSS'} — {symbol.replace('NSE:','').replace('-EQ','').replace('-INDEX','')}",
            body=(
                f"Exit   : ₹{price:,.2f}  (qty {qty})\n"
                f"P&L    : ₹{pnl:+,.0f}  ({pnl_pct:+.2f}%)\n"
                f"Reason : {reason}"
            ),
        )
        self._dispatch(n)

    def notify_circuit_break(self, reason: str, detail: str = "") -> None:
        n = Notification(
            event="CIRCUIT",
            emoji="🚨",
            title="CIRCUIT BREAKER TRIPPED — Trading Halted",
            body=f"Reason : {reason}\n{detail}\nManual reset required.",
        )
        self._dispatch(n)

    def notify_daily_summary(
        self,
        trades: int,
        win_rate: float,
        total_pnl: float,
        total_pnl_pct: float,
        max_dd: float,
    ) -> None:
        icon = "🟢" if total_pnl >= 0 else "🔴"
        n = Notification(
            event="DAILY",
            emoji=icon,
            title="Daily Summary — NSE Algo Paper Trading",
            body=(
                f"Trades   : {trades}\n"
                f"Win Rate : {win_rate:.1f}%\n"
                f"P&L      : ₹{total_pnl:+,.0f}  ({total_pnl_pct:+.2f}%)\n"
                f"Max DD   : {max_dd:.2f}%"
            ),
        )
        self._dispatch(n)

    # ── Test ──────────────────────────────────────────────────────────────────

    def test(self) -> dict[str, bool]:
        """Send a test notification on all channels. Returns {channel: success}."""
        n = Notification(
            event="TEST",
            emoji="🔔",
            title="NSE Algo Trader — Test Notification",
            body="All notification channels are working correctly.",
        )
        results = {}
        results["telegram"] = self._send_telegram(n)
        results["desktop"]  = self._send_desktop(n)
        self._write_toast(n)
        results["dashboard"] = True
        return results

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(self, n: Notification) -> None:
        logger.info("NOTIFY [{}] {}", n.event, n.title)
        self._send_telegram(n)
        self._send_desktop(n)
        self._write_toast(n)

    # ── Telegram ──────────────────────────────────────────────────────────────

    def _send_telegram(self, n: Notification) -> bool:
        if not self.telegram_token or not self.telegram_chat:
            return False
        try:
            import urllib.request, json as _json
            text = f"<b>{n.emoji} {n.title}</b>\n<code>{n.time} IST</code>\n\n{n.body}"
            payload = _json.dumps({
                "chat_id":    self.telegram_chat,
                "text":       text,
                "parse_mode": "HTML",
            }).encode()
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            logger.debug("Telegram notification sent ✓")
            return True
        except Exception as e:
            logger.debug("Telegram notification failed: {}", e)
            return False

    # ── macOS desktop ─────────────────────────────────────────────────────────

    def _send_desktop(self, n: Notification) -> bool:
        if not self.desktop_enabled:
            return False
        try:
            # Single-line body for osascript
            body = n.body.replace("\n", "  |  ").replace("'", "")
            title = n.title.replace("'", "")
            script = (
                f"display notification '{body}' "
                f"with title 'NSE Algo Trader' "
                f"subtitle '{title}' "
                f"sound name 'Glass'"
            )
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, timeout=3,
            )
            return True
        except Exception:
            return False

    # ── Streamlit toast (written to state file) ───────────────────────────────

    def _write_toast(self, n: Notification) -> None:
        try:
            NOTIF_FILE.parent.mkdir(parents=True, exist_ok=True)
            NOTIF_FILE.write_text(json.dumps({
                "event":   n.event,
                "emoji":   n.emoji,
                "title":   n.title,
                "body":    n.body,
                "time":    n.time,
                "read":    False,
            }))
        except Exception:
            pass


# ── Singleton ─────────────────────────────────────────────────────────────────

_notifier: NotificationManager | None = None

def get_notifier() -> NotificationManager:
    global _notifier
    if _notifier is None:
        _notifier = NotificationManager()
    return _notifier
