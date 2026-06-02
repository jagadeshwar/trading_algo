"""Model training script — Phase 2 XGBoost + HMM.

Trains and saves the two Phase 2 ML models:
  1. XGBoost direction classifier  → models/xgb_direction.pkl
  2. HMM regime detector           → models/regime_hmm.pkl

Then runs walk-forward validation with DSR gating to confirm live-readiness.

Usage:
    python train_models.py                              # Nifty50 5min (default)
    python train_models.py --symbol NSE:NIFTYBANK-INDEX --interval 5
    python train_models.py --symbol NSE:NIFTY50-INDEX  --skip-wfv
    python train_models.py --xgb-only
    python train_models.py --hmm-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


def _load_features(symbol: str, interval_min: int):
    import pyarrow.parquet as pq
    import pandas as pd

    slug       = symbol.replace(":", "_").replace("-", "_")
    feat_path  = Path(f"data/features/{slug}_{interval_min}min_features.parquet")
    ohlcv_path = Path(f"data/ohlcv/{slug}_{interval_min}min.parquet")

    if not feat_path.exists():
        logger.error("Feature file not found: {}", feat_path)
        logger.info("Run the data fetcher first:  python run_paper_trading.py --backtest  or DataFetcher.incremental_update()")
        sys.exit(1)

    feats = pq.read_table(feat_path).to_pandas()
    feats.index = pd.to_datetime(feats.index, utc=True).tz_convert("Asia/Kolkata")
    logger.info("Loaded {} feature rows for {} {}min  [{} → {}]",
                len(feats), symbol, interval_min,
                feats.index.min().date(), feats.index.max().date())

    if ohlcv_path.exists():
        ohlcv = pq.read_table(ohlcv_path).to_pandas()
        ohlcv.index = pd.to_datetime(ohlcv.index, utc=True).tz_convert("Asia/Kolkata")
        feats["close"] = ohlcv["close"].reindex(feats.index)

    return feats


def train_hmm(feats, save_path: Path = Path("models/regime_hmm.pkl")) -> None:
    from production.models.regime_hmm import RegimeDetector

    logger.info("=== Training HMM regime detector ===")
    det = RegimeDetector(n_states=3, n_iter=200)
    det.fit(feats)
    det.save(save_path)

    labels = det.predict(feats)
    counts = labels.value_counts().sort_index()
    for state, count in counts.items():
        pct = 100 * count / len(labels)
        logger.info("  State {:>12}: {:>5} bars ({:.1f}%)", state, count, pct)


def train_xgb(feats, save_path: Path = Path("models/xgb_direction.pkl"),
              n_estimators: int = 500, max_depth: int = 4,
              learning_rate: float = 0.05, subsample: float = 0.8) -> None:
    from production.models.xgb_classifier import DirectionClassifier

    logger.info("=== Training XGBoost direction classifier ===")
    if "target" not in feats.columns or feats["target"].isna().all():
        logger.error("Feature data has no 'target' column — cannot train XGBoost")
        return

    clf = DirectionClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        min_gap=20,
        n_splits=5,
    )
    clf.fit(feats)
    clf.save(save_path)

    cv = clf.cv_results()
    logger.info("CV accuracy: {:.3f}", cv.get("mean_accuracy", cv.get("accuracy", 0)))

    top = clf.feature_importance().head(10)
    logger.info("Top-10 features by importance:")
    for feat, imp in top.items():
        logger.info("  {:>30}  {:.4f}", feat, imp)


def run_wfv(symbol: str, interval_min: int, dsr_threshold: float = 1.0) -> bool:
    from production.strategy.walk_forward import WalkForwardValidator

    logger.info("=== Walk-Forward Validation (DSR gate: {:.1f}) ===", dsr_threshold)
    wfv    = WalkForwardValidator(dsr_threshold=dsr_threshold)
    result = wfv.run(symbol=symbol, interval_min=interval_min)
    logger.info(result.summary())

    for i, w in enumerate(result.window_stats, 1):
        logger.info("  Window {:>2}: sharpe={:.2f}  dd={:.2f}%  trades={:>3}  pf={:.2f}",
                    i, w.get("sharpe", 0), w.get("max_dd_pct", 0),
                    w.get("trades", 0), w.get("profit_factor", 0))

    return result.live_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="NSE Algo — Train Phase 2 ML Models")
    parser.add_argument("--symbol",       default="NSE:NIFTY50-INDEX")
    parser.add_argument("--interval",     default=5, type=int)
    parser.add_argument("--xgb-only",     action="store_true", help="Train XGBoost only")
    parser.add_argument("--hmm-only",     action="store_true", help="Train HMM only")
    parser.add_argument("--skip-wfv",     action="store_true", help="Skip walk-forward validation")
    parser.add_argument("--dsr-threshold", default=1.0, type=float)
    parser.add_argument("--n-estimators", default=500, type=int)
    parser.add_argument("--max-depth",    default=4, type=int)
    parser.add_argument("--learning-rate", default=0.05, type=float)
    parser.add_argument("--subsample",    default=0.8, type=float)
    args = parser.parse_args()

    feats = _load_features(args.symbol, args.interval)
    xgb_path = Path("models/xgb_direction.pkl")
    hmm_path = Path("models/regime_hmm.pkl")

    do_hmm = not args.xgb_only
    do_xgb = not args.hmm_only

    if do_hmm:
        train_hmm(feats, save_path=hmm_path)
        # Re-load HMM and recompute regime_hmm feature so XGB trains with updated labels
        from production.data.features import FeatureEngineer
        import pyarrow.parquet as pq
        slug      = args.symbol.replace(":", "_").replace("-", "_")
        ohlcv_path = Path(f"data/ohlcv/{slug}_{args.interval}min.parquet")
        if ohlcv_path.exists():
            ohlcv = pq.read_table(ohlcv_path).to_pandas()
            import pandas as pd
            ohlcv.index = pd.to_datetime(ohlcv.index, utc=True).tz_convert("Asia/Kolkata")
            feats = FeatureEngineer().compute(ohlcv, hmm_model_path=str(hmm_path))
            feats["close"] = ohlcv["close"].reindex(feats.index)
            logger.info("Recomputed features with updated HMM labels ({} bars)", len(feats))

    if do_xgb:
        train_xgb(feats, save_path=xgb_path,
                  n_estimators=args.n_estimators,
                  max_depth=args.max_depth,
                  learning_rate=args.learning_rate,
                  subsample=args.subsample)

    if not args.skip_wfv:
        live_ready = run_wfv(args.symbol, args.interval, dsr_threshold=args.dsr_threshold)
        if live_ready:
            logger.info("✅ Models validated — DSR gate PASSED. System is live-ready.")
        else:
            logger.warning("❌ DSR gate FAILED. Do not go live without further validation.")
            sys.exit(2)
    else:
        logger.info("Walk-forward skipped (--skip-wfv). Run without --skip-wfv before going live.")


if __name__ == "__main__":
    main()
