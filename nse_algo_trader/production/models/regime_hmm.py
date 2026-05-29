"""HMM Regime Detector — classifies market into 3 states.

States (auto-labelled after fitting):
  0 → RANGING    (low volatility, no clear trend)
  1 → TRENDING   (sustained directional move)
  2 → HIGH_VOL   (elevated volatility, mean-reverting)

Usage:
    from production.models.regime_hmm import RegimeDetector
    det = RegimeDetector()
    det.fit(features_df)           # train on historical features
    det.save("models/regime_hmm.pkl")

    labels = det.predict(features_df)   # returns Series of 0/1/2
    det.load("models/regime_hmm.pkl")
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger


# ── Observation features used to train the HMM ────────────────────────────────
_OBS_COLS = [
    "ret_1b",       # 1-bar return (direction signal)
    "atr_pct",      # ATR % (volatility proxy)
    "adx",          # trend strength
    "vol_ratio",    # volume relative to MA (activity)
]

_STATE_NAMES = ["RANGING", "TRENDING", "HIGH_VOL"]


class RegimeDetector:
    """Gaussian HMM with 3 hidden states trained on price + vol features."""

    def __init__(self, n_states: int = 3, n_iter: int = 200, random_state: int = 42) -> None:
        self.n_states     = n_states
        self.n_iter       = n_iter
        self.random_state = random_state
        self._model: GaussianHMM | None = None
        self._state_map: dict[int, str] = {}   # HMM state idx → regime name
        self._fitted = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame) -> "RegimeDetector":
        obs, lengths = self._prepare(features)
        logger.info("Training HMM on {} bars × {} obs features", len(obs), obs.shape[1])

        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._model.fit(obs, lengths)
        self._label_states()
        self._fitted = True
        logger.info("HMM trained — state map: {}", self._state_map)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return a Series of regime labels (0=RANGING, 1=TRENDING, 2=HIGH_VOL)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        obs, _ = self._prepare(features)
        raw    = self._model.predict(obs)
        mapped = np.vectorize(self._state_map.get)(raw)
        result = pd.Series(mapped, index=features.index[-len(obs):], name="regime_hmm")
        return result

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return state posterior probabilities (n_bars × 3)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        obs, _ = self._prepare(features)
        posteriors = self._model.predict_proba(obs)
        cols = [self._state_map.get(i, str(i)) for i in range(self.n_states)]
        return pd.DataFrame(posteriors, index=features.index[-len(obs):], columns=cols)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "state_map": self._state_map}, f)
        logger.info("HMM saved → {}", path)

    def load(self, path: str | Path) -> "RegimeDetector":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model    = data["model"]
        self._state_map = data["state_map"]
        self._fitted   = True
        logger.info("HMM loaded ← {}", path)
        return self

    # ── Internals ─────────────────────────────────────────────────────────────

    def _prepare(self, features: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
        cols = [c for c in _OBS_COLS if c in features.columns]
        if not cols:
            raise ValueError(f"None of {_OBS_COLS} found in features")
        df  = features[cols].dropna()
        obs = df.values.astype(np.float64)
        # Standardise each column to zero mean, unit std
        obs = (obs - obs.mean(axis=0)) / (obs.std(axis=0) + 1e-9)
        return obs, [len(obs)]

    def _label_states(self) -> None:
        """Auto-label each HMM state based on its mean observation vector."""
        means = self._model.means_                 # (n_states, n_obs_features)
        cols  = [c for c in _OBS_COLS]
        ret_idx = cols.index("ret_1b")   if "ret_1b"   in cols else None
        atr_idx = cols.index("atr_pct")  if "atr_pct"  in cols else None
        adx_idx = cols.index("adx")      if "adx"      in cols else None

        # Rank states by ADX mean → highest ADX = TRENDING
        if adx_idx is not None:
            adx_means = means[:, adx_idx]
            ranked    = np.argsort(adx_means)   # ascending
        else:
            ranked = np.arange(self.n_states)

        # Rank by ATR among the bottom-ADX states → highest ATR = HIGH_VOL
        if atr_idx is not None and self.n_states >= 3:
            low_adx_states = ranked[:2]
            atr_vals       = means[low_adx_states, atr_idx]
            high_vol_idx   = low_adx_states[np.argmax(atr_vals)]
            ranging_idx    = low_adx_states[np.argmin(atr_vals)]
            trending_idx   = ranked[-1]
        else:
            ranging_idx  = ranked[0]
            trending_idx = ranked[-1]
            high_vol_idx = ranked[1] if self.n_states >= 3 else ranked[0]

        self._state_map = {
            int(ranging_idx):  0,   # RANGING
            int(trending_idx): 1,   # TRENDING
            int(high_vol_idx): 2,   # HIGH_VOL
        }
