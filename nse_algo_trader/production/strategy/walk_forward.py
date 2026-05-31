"""Walk-Forward Validation with Deflated Sharpe Ratio — Phase 2.

Splits the historical data into expanding train/test windows, runs the
full strategy pipeline (features → XGBoost signal → paper fills) on each
test window, then aggregates performance metrics across all windows.

DSR gate: strategy is 'live-ready' only if DSR > 1.0 on the OOS data.

Usage:
    from production.strategy.walk_forward import WalkForwardValidator
    wfv = WalkForwardValidator()
    result = wfv.run(symbol="NSE:NIFTY50-INDEX", interval_min=5)
    print(result["live_ready"], result["dsr"], result["sharpe"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger


@dataclass
class WFVResult:
    symbol:        str
    n_windows:     int
    sharpe:        float
    dsr:           float
    win_rate:      float
    profit_factor: float
    total_trades:  int
    total_pnl_pct: float
    max_dd_pct:    float
    live_ready:    bool
    window_stats:  list[dict] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ LIVE-READY" if self.live_ready else "❌ NOT READY"
        return (
            f"{status}  DSR={self.dsr:.2f}  Sharpe={self.sharpe:.2f}  "
            f"WinRate={self.win_rate:.1%}  PF={self.profit_factor:.2f}  "
            f"Trades={self.total_trades}  MaxDD={self.max_dd_pct:.2f}%"
        )


class WalkForwardValidator:
    """Expanding-window walk-forward backtest with DSR gating."""

    def __init__(
        self,
        train_bars:  int   = 2000,   # minimum training bars per window
        test_bars:   int   = 500,    # test bars per window
        n_windows:   int   = 5,      # number of walk-forward windows
        capital:     float = 500_000,
        cost_bps:    float = 4.5,    # all-in transaction cost (bps)
        dsr_threshold: float = 1.0,
    ) -> None:
        self.train_bars    = train_bars
        self.test_bars     = test_bars
        self.n_windows     = n_windows
        self.capital       = capital
        self.cost_bps      = cost_bps
        self.dsr_threshold = dsr_threshold

    # ── Public ─────────────────────────────────────────────────────────────────

    def run(self, symbol: str, interval_min: int = 5) -> WFVResult:
        feats = self._load_features(symbol, interval_min)
        if feats is None or len(feats) < self.train_bars + self.test_bars:
            logger.warning("Insufficient data for walk-forward: {} bars", len(feats) if feats is not None else 0)
            return self._empty_result(symbol)

        logger.info("Walk-forward: {} | {} bars total", symbol, len(feats))

        windows      = self._make_windows(feats)
        window_stats = []
        all_rets     = []

        for i, (train_df, test_df) in enumerate(windows):
            logger.info("Window {}/{}: train={} test={}", i + 1, len(windows), len(train_df), len(test_df))
            stats = self._run_window(train_df, test_df, i)
            window_stats.append(stats)
            all_rets.extend(stats["bar_returns"])

        return self._aggregate(symbol, window_stats, all_rets)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _load_features(self, symbol: str, interval_min: int) -> pd.DataFrame | None:
        from production.data.features import FeatureEngineer
        slug       = symbol.replace(":", "_").replace("-", "_")
        ohlcv_path = Path(f"data/ohlcv/{slug}_{interval_min}min.parquet")
        if not ohlcv_path.exists():
            logger.error("OHLCV not found: {}", ohlcv_path)
            return None
        ohlcv = pq.read_table(ohlcv_path).to_pandas()
        ohlcv.index = pd.to_datetime(ohlcv.index, utc=True).tz_convert("Asia/Kolkata")
        hmm_path = str(Path("models/regime_hmm.pkl")) if Path("models/regime_hmm.pkl").exists() else None
        feats = FeatureEngineer().compute(ohlcv, hmm_model_path=hmm_path)
        # Merge raw close price so walk-forward can simulate fills
        feats["close"] = ohlcv["close"].reindex(feats.index)
        return feats

    def _make_windows(self, feats: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        total = len(feats)
        step  = self.test_bars
        windows = []
        for i in range(self.n_windows):
            test_end   = total - (self.n_windows - 1 - i) * step
            test_start = test_end - step
            train_end  = test_start
            if train_end < self.train_bars:
                continue
            train_df = feats.iloc[:train_end]
            test_df  = feats.iloc[test_start:test_end]
            windows.append((train_df, test_df))
        return windows

    def _run_window(self, train_df: pd.DataFrame, test_df: pd.DataFrame, fold: int) -> dict:
        """Train XGBoost on train_df, simulate trades on test_df."""
        from production.models.xgb_classifier import DirectionClassifier
        from production.strategy.momentum import MomentumStrategy, load_config

        # Train fresh XGB on this window's training data
        clf = DirectionClassifier(n_estimators=200, n_splits=3)
        try:
            clf.fit(train_df)
        except Exception as e:
            logger.warning("XGB fit failed on window {}: {}", fold, e)
            clf = None

        strategy = MomentumStrategy(load_config())
        if clf:
            strategy._xgb_clf = clf

        capital   = self.capital
        equity    = capital
        positions: dict[str, dict] = {}
        bar_returns = []
        trades = []

        for ts, row in test_df.iterrows():
            close = float(row.get("close", 0)) if "close" in row.index else 0
            if close == 0:
                # features df may not have raw close — use vwap_dev proxy
                bar_returns.append(0.0)
                continue

            # Check exits first
            for sym in list(positions.keys()):
                pos   = positions[sym]
                dirn  = pos["direction"]
                entry = pos["entry"]
                stop  = pos["stop"]
                tgt   = pos["target"]
                if (dirn == 1 and (close <= stop or close >= tgt)) or \
                   (dirn == -1 and (close >= stop or close <= tgt)):
                    pnl = dirn * (close - entry) * pos["qty"]
                    costs = entry * pos["qty"] * self.cost_bps / 10_000
                    net_pnl = pnl - costs
                    equity += net_pnl
                    trades.append(net_pnl)
                    del positions[sym]

            # New signal
            if not positions:
                sig = strategy.evaluate_bar("SIM", row, close)
                if sig and sig.confidence >= 0.6:
                    qty = max(1, int(equity * 0.05 / (abs(close - sig.stop_price) + 1e-6)))
                    qty = min(qty, 5)
                    positions["SIM"] = {
                        "direction": sig.direction,
                        "entry":     close,
                        "stop":      sig.stop_price,
                        "target":    sig.target_price,
                        "qty":       qty,
                    }

            bar_returns.append(equity)   # track running equity level (not fraction)

        equity_series = pd.Series(bar_returns)
        bar_rets      = equity_series.pct_change().dropna()
        sharpe        = float(bar_rets.mean() / (bar_rets.std() + 1e-9) * np.sqrt(252 * 78))

        wins   = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        pf     = sum(wins) / (abs(sum(losses)) + 1e-9) if losses else float("inf")

        return {
            "fold":         fold,
            "n_trades":     len(trades),
            "win_rate":     len(wins) / len(trades) if trades else 0.0,
            "profit_factor": pf,
            "pnl_pct":      (equity - self.capital) / self.capital * 100,
            "sharpe":       sharpe,
            "bar_returns":  bar_returns,    # equity levels, not fractions
        }

    def _aggregate(self, symbol: str, window_stats: list[dict], all_rets: list[float]) -> WFVResult:
        from production.models.purged_cv import deflated_sharpe_ratio

        rets      = pd.Series(all_rets).pct_change().dropna()  # equity levels → pct returns
        sharpe    = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252 * 78))
        skew      = float(rets.skew())
        kurt      = float(rets.kurtosis())
        n_trials  = self.n_windows
        n_obs     = len(rets)

        dsr = deflated_sharpe_ratio(sharpe, n_trials=n_trials, n_obs=n_obs,
                                    skewness=skew, kurtosis=kurt)

        all_trades    = sum(w["n_trades"] for w in window_stats)
        avg_win_rate  = np.mean([w["win_rate"] for w in window_stats if w["n_trades"] > 0])
        avg_pf        = np.mean([w["profit_factor"] for w in window_stats if w["n_trades"] > 0])
        total_pnl_pct = sum(w["pnl_pct"] for w in window_stats)

        # Max drawdown — all_rets are equity levels now
        equity_curve = pd.Series(all_rets)
        roll_max     = equity_curve.cummax()
        drawdown     = (equity_curve - roll_max) / roll_max * 100
        max_dd       = float(drawdown.min())

        result = WFVResult(
            symbol        = symbol,
            n_windows     = len(window_stats),
            sharpe        = round(sharpe, 3),
            dsr           = round(dsr, 3),
            win_rate      = round(float(avg_win_rate), 3),
            profit_factor = round(float(avg_pf), 3),
            total_trades  = all_trades,
            total_pnl_pct = round(total_pnl_pct, 3),
            max_dd_pct    = round(abs(max_dd), 3),
            live_ready    = dsr >= self.dsr_threshold,
            window_stats  = window_stats,
        )
        logger.info("Walk-forward complete: {}", result.summary())
        return result

    def _empty_result(self, symbol: str) -> WFVResult:
        return WFVResult(symbol=symbol, n_windows=0, sharpe=0.0, dsr=0.0,
                         win_rate=0.0, profit_factor=0.0, total_trades=0,
                         total_pnl_pct=0.0, max_dd_pct=0.0, live_ready=False)
