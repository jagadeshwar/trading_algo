"""XGBoost Direction Classifier — Phase 2 ML signal.

Predicts 3-bar-ahead direction: 1=Up, 0=Flat, -1=Down.
Validated with PurgedKFold to prevent look-ahead leakage.
Replaces the pure momentum signal in paper trading.

Usage:
    from production.models.xgb_classifier import DirectionClassifier
    clf = DirectionClassifier()
    clf.fit(features_df)
    clf.save("models/xgb_direction.pkl")

    signal = clf.predict_bar(latest_features_series)
    # returns {"direction": 1/-1/0, "confidence": 0.72, "proba": {...}}
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# XGBoost feature columns (all except target + regime_heuristic which is noisy)
_EXCL = {"target", "regime_heuristic"}

_MODEL_PATH = Path("models/xgb_direction.pkl")


class DirectionClassifier:
    """XGBoost 3-class direction model with PurgedKFold CV."""

    def __init__(
        self,
        n_estimators:  int   = 500,
        max_depth:     int   = 4,
        learning_rate: float = 0.05,
        subsample:     float = 0.8,
        colsample:     float = 0.8,
        min_gap:       int   = 20,   # PurgedKFold embargo bars
        n_splits:      int   = 5,
        random_state:  int   = 42,
    ) -> None:
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self.colsample     = colsample
        self.min_gap       = min_gap
        self.n_splits      = n_splits
        self.random_state  = random_state
        self._model        = None
        self._features:    list[str] = []
        self._cv_results:  dict      = {}
        self._fitted       = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame) -> "DirectionClassifier":
        import xgboost as xgb
        from production.models.purged_cv import PurgedKFold

        X, y, df_clean = self._prepare(features)
        logger.info("XGB training on {} bars × {} features", len(X), X.shape[1])

        # ── PurgedKFold cross-validation ──────────────────────────────────────
        cv     = PurgedKFold(n_splits=self.n_splits, embargo_pct=self.min_gap / len(X))
        scores = {"accuracy": []}

        for fold, (tr_idx, val_idx) in enumerate(cv.split(df_clean)):
            X_tr, y_tr   = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            m = self._make_model()
            m.fit(X_tr, y_tr)

            preds_enc = m.predict(X_val)
            preds = self._decode(preds_enc)
            y_dec = self._decode(y_val)
            acc   = (preds == y_dec).mean()
            scores["accuracy"].append(acc)
            logger.info("Fold {}: accuracy={:.3f}", fold + 1, acc)

        logger.info("CV accuracy: {:.3f} ± {:.3f}",
                    np.mean(scores["accuracy"]), np.std(scores["accuracy"]))
        self._cv_results = {k: float(np.mean(v)) for k, v in scores.items()}

        # ── Final model on full data ──────────────────────────────────────────
        self._model = self._make_model()
        self._model.fit(X, y)   # y is encoded [0,1,2]
        self._fitted = True
        logger.info("XGB final model trained — features: {}", len(self._features))
        return self

    def predict_bar(self, row: pd.Series) -> dict:
        """Predict direction for a single feature row. Returns dict with direction + confidence."""
        if not self._fitted:
            raise RuntimeError("Call fit() or load() first")
        x     = self._row_to_array(row)
        proba = self._model.predict_proba(x)[0]   # [P(down), P(flat), P(up)] encoded order
        enc_pred   = int(self._model.predict(x)[0])
        direction  = self._decode(np.array([enc_pred]))[0]
        confidence = float(max(proba))
        # Map encoded class idx back to -1/0/1 for proba dict
        proba_map = {self._decode(np.array([i]))[0]: float(p)
                     for i, p in enumerate(proba)}
        return {
            "direction":  int(direction),
            "confidence": confidence,
            "proba":      proba_map,
        }

    def feature_importance(self) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        imp = self._model.feature_importances_
        return pd.Series(imp, index=self._features).sort_values(ascending=False)

    def cv_results(self) -> dict:
        return self._cv_results.copy()

    def save(self, path: str | Path = _MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":      self._model,
                "features":   self._features,
                "cv_results": self._cv_results,
            }, f)
        logger.info("XGB model saved → {}", path)

    def load(self, path: str | Path = _MODEL_PATH) -> "DirectionClassifier":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model      = data["model"]
        self._features   = data["features"]
        self._cv_results = data.get("cv_results", {})
        self._fitted     = True
        logger.info("XGB model loaded ← {} | features: {}", path, len(self._features))
        return self

    # ── Internals ─────────────────────────────────────────────────────────────

    # XGBoost needs labels in [0, num_class): map -1→0, 0→1, 1→2
    _ENCODE = {-1: 0, 0: 1, 1: 2}
    _DECODE = {0: -1, 1: 0, 2: 1}

    def _encode(self, y: np.ndarray) -> np.ndarray:
        return np.vectorize(self._ENCODE.get)(y).astype(np.int32)

    def _decode(self, y: np.ndarray) -> np.ndarray:
        return np.vectorize(self._DECODE.get)(y).astype(np.int32)

    def _prepare(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        cols = [c for c in features.columns if c not in _EXCL and c != "target"]
        self._features = cols
        df = features[cols + ["target"]].dropna()
        X  = df[cols].values.astype(np.float32)
        y  = self._encode(df["target"].values.astype(np.int32))
        return X, y, df

    def _row_to_array(self, row: pd.Series) -> np.ndarray:
        vals = [float(row.get(f, 0.0)) for f in self._features]
        return np.array([vals], dtype=np.float32)

    def _make_model(self):
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators    = self.n_estimators,
            max_depth       = self.max_depth,
            learning_rate   = self.learning_rate,
            subsample       = self.subsample,
            colsample_bytree= self.colsample,
            eval_metric     = "mlogloss",
            random_state    = self.random_state,
            n_jobs          = -1,
            tree_method     = "hist",
        )
