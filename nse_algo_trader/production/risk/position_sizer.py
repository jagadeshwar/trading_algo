"""Kelly criterion position sizer with risk parameter enforcement."""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger


def _load_risk_cfg(path: str = "configs/risk.yaml") -> dict:
    try:
        return yaml.safe_load(Path(path).read_text())
    except Exception:
        return {}


class PositionSizer:
    """Half-Kelly position sizing with max-position and margin guards.

    All sizes are returned as number of shares/lots. The caller is responsible
    for rounding to the correct NSE lot size.
    """

    def __init__(self, capital: float, config_path: str = "configs/risk.yaml") -> None:
        self.capital = capital
        cfg = _load_risk_cfg(config_path)
        self.kelly_fraction   = cfg.get("kelly_fraction", 0.5)
        self.max_position_pct = cfg.get("max_position_pct", 10.0) / 100.0
        self.max_capital_use  = cfg.get("max_capital_use",  60.0) / 100.0
        self.daily_loss_pct   = cfg.get("daily_loss_pct",    2.0) / 100.0

        # Running stats updated after each trade — seeded with conservative defaults
        self._win_rate:   float = 0.52
        self._avg_win:    float = 1.5   # R multiples
        self._avg_loss:   float = 1.0   # R multiples
        self._trade_count: int  = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def size(
        self,
        confidence: float,
        entry_price: float,
        stop_price: float,
        current_capital: float | None = None,
        current_exposure: float = 0.0,
    ) -> int:
        """Return number of shares to buy/sell.

        Parameters
        ----------
        confidence       : signal confidence 0–1 (scales Kelly fraction)
        entry_price      : expected fill price
        stop_price       : stop-loss price (defines risk per share)
        current_capital  : live capital (uses self.capital if None)
        current_exposure : current open position value (for margin guard)
        """
        capital = current_capital if current_capital is not None else self.capital
        risk_per_share = abs(entry_price - stop_price)

        if risk_per_share <= 0 or entry_price <= 0:
            return 0

        # ── Full Kelly ────────────────────────────────────────────────────────
        reward_risk = (abs(entry_price - self._estimated_target(entry_price, stop_price))
                       / risk_per_share)
        full_kelly = self._kelly(self._win_rate, reward_risk)
        if full_kelly <= 0:
            return 0

        # ── Scale by confidence and fraction ─────────────────────────────────
        scaled_kelly = full_kelly * self.kelly_fraction * confidence
        risk_capital = capital * scaled_kelly

        # ── Max position cap ──────────────────────────────────────────────────
        max_position_value = capital * self.max_position_pct
        position_value = min(risk_capital / (risk_per_share / entry_price),
                             max_position_value)

        # ── Margin guard: don't exceed 60% total exposure ─────────────────────
        available_exposure = capital * self.max_capital_use - current_exposure
        if available_exposure <= 0:
            logger.warning("PositionSizer: margin cap reached, blocking new position")
            return 0
        position_value = min(position_value, available_exposure)

        shares = int(position_value / entry_price)
        if shares < 1:
            return 0

        logger.debug(
            "PositionSizer: capital={:.0f} kelly={:.3f} conf={:.2f} → {} shares "
            "(value={:.0f}, risk/share={:.2f})",
            capital, scaled_kelly, confidence, shares, position_value, risk_per_share,
        )
        return shares

    def update_stats(self, pnl: float, risk: float) -> None:
        """Call after each closed trade to update win rate and R stats."""
        self._trade_count += 1
        r = pnl / risk if risk > 0 else 0
        # Exponential moving average (α=0.1) — recent trades weight more
        alpha = 0.1
        if pnl > 0:
            self._win_rate  = (1 - alpha) * self._win_rate + alpha * 1.0
            self._avg_win   = (1 - alpha) * self._avg_win  + alpha * r
        else:
            self._win_rate  = (1 - alpha) * self._win_rate + alpha * 0.0
            self._avg_loss  = (1 - alpha) * self._avg_loss + alpha * abs(r)

        logger.debug(
            "PositionSizer stats updated: win_rate={:.3f} avg_win={:.2f}R avg_loss={:.2f}R (n={})",
            self._win_rate, self._avg_win, self._avg_loss, self._trade_count,
        )

    @property
    def stats(self) -> dict:
        return {
            "win_rate":    round(self._win_rate, 4),
            "avg_win_r":  round(self._avg_win, 3),
            "avg_loss_r": round(self._avg_loss, 3),
            "trade_count": self._trade_count,
            "full_kelly":  round(self._kelly(self._win_rate, self._avg_win / max(self._avg_loss, 0.01)), 4),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _kelly(win_rate: float, reward_risk: float) -> float:
        if reward_risk <= 0 or win_rate <= 0:
            return 0.0
        return max(0.0, win_rate - (1 - win_rate) / reward_risk)

    def _estimated_target(self, entry: float, stop: float) -> float:
        risk = abs(entry - stop)
        direction = 1 if entry > stop else -1
        return entry + direction * risk * 3.0   # 3× ATR target (from strategy config)
