"""PurgedKFold cross-validation and Deflated Sharpe Ratio.

Implements the techniques from Lopez de Prado's
"Advances in Financial Machine Learning" (2018).

These replace the mlfinlab dependency (no longer on PyPI).
"""

from __future__ import annotations

import math
from typing import Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# ── PurgedKFold ───────────────────────────────────────────────────────────────

class PurgedKFold:
    """K-Fold cross-validator that removes overlapping observations.

    Standard KFold leaks future information in financial time series because
    training and test samples share the same forward-looking label window.
    PurgedKFold removes ("purges") any training observation whose label
    period overlaps with the test fold's bar period, then adds an embargo
    gap after the test fold to prevent look-ahead via autocorrelation.

    Parameters
    ----------
    n_splits:        Number of folds (k).
    embargo_pct:     Fraction of total samples to embargo after each test fold.
                     Lopez de Prado recommends 0.01–0.02 (1–2% of dataset).
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        pred_times: pd.Series | None = None,
        eval_times: pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (train_idx, test_idx) arrays.

        Parameters
        ----------
        X:            Feature DataFrame with DatetimeIndex.
        pred_times:   Series of prediction (bar) timestamps, same index as X.
                      Defaults to X.index.
        eval_times:   Series of evaluation (label end) timestamps, same index.
                      Defaults to pred_times + target_bars (if known) or pred_times.
        """
        indices = np.arange(len(X))
        embargo_size = int(len(X) * self.embargo_pct)

        if pred_times is None:
            pred_times = pd.Series(X.index, index=X.index)
        if eval_times is None:
            eval_times = pred_times

        kf = KFold(n_splits=self.n_splits)
        for train_idx, test_idx in kf.split(indices):
            test_start = pred_times.iloc[test_idx[0]]
            test_end = eval_times.iloc[test_idx[-1]]

            # Purge: remove training samples whose eval_time overlaps with test window
            purge_mask = eval_times.iloc[train_idx] >= test_start
            purged_train = train_idx[~purge_mask.values]

            # Embargo: remove training samples immediately after the test fold
            if embargo_size > 0 and test_idx[-1] + 1 < len(X):
                embargo_end = min(test_idx[-1] + embargo_size, len(X) - 1)
                embargo_mask = (pred_times.iloc[purged_train] > test_end) & (
                    pred_times.iloc[purged_train]
                    <= pred_times.iloc[embargo_end]
                )
                purged_train = purged_train[~embargo_mask.values]

            yield purged_train, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ── Deflated Sharpe Ratio ─────────────────────────────────────────────────────

def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (DSR).

    Corrects the observed Sharpe Ratio for multiple-testing bias.
    If you ran N strategy variants in your search, the best Sharpe
    will be inflated by luck. DSR deflates it by the expected maximum
    Sharpe under the null hypothesis (no skill).

    Parameters
    ----------
    sharpe:    Annualised Sharpe ratio of the best strategy found.
    n_trials:  Number of strategies / parameter combinations tested.
    n_obs:     Number of return observations used to compute Sharpe.
    skewness:  Skewness of the strategy returns (default 0 = normal).
    kurtosis:  Excess kurtosis of returns (default 3 = normal). Note: pass
               the raw kurtosis value, not excess kurtosis.

    Returns
    -------
    DSR value. Go live only if DSR > 1.0 (conventional threshold).

    References
    ----------
    Lopez de Prado & Bailey, "The Deflated Sharpe Ratio" (2014).
    """
    from scipy.stats import norm

    # Expected maximum Sharpe under no-skill null (Bailey & Lopez de Prado 2014)
    gamma = 0.5772156649  # Euler-Mascheroni constant
    expected_max = (1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(
        1 - 1 / (n_trials * math.e)
    )

    # Adjust for non-normality of returns (higher moments deflate further)
    variance_sr = (
        1
        - skewness * sharpe
        + ((kurtosis - 1) / 4) * sharpe**2
    ) / (n_obs - 1)

    # Probabilistic Sharpe Ratio — probability the true SR > benchmark SR
    if variance_sr <= 0:
        return 0.0
    psr = norm.cdf((sharpe - expected_max) / math.sqrt(variance_sr))
    return psr


def is_live_ready(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    dsr_threshold: float = 0.95,
) -> tuple[bool, float]:
    """Return (ready_to_go_live, dsr_value).

    dsr_threshold of 0.95 means 95% probability the Sharpe is genuinely > 0.
    """
    dsr = deflated_sharpe_ratio(sharpe, n_trials, n_obs, skewness, kurtosis)
    return dsr >= dsr_threshold, dsr
