"""
Distributional Similarity Index (DSI).

Fits a Gaussian Mixture Model on the numerical columns of the real data, then
compares per-row log-likelihoods of synthetic rows against the average per-row
log-likelihood of real rows.

    DSI = (1 / K) * sum_{i=1}^K (1 - |log L(x_syn,i) - L_real| / |L_real|)

where L_real is the average per-row log-likelihood of the real data under the
fitted GMM and log L(x_syn,i) is the log-likelihood of the i-th synthetic row
under the same model.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def _numeric_matrix(df: pd.DataFrame, cols: Iterable[str]) -> np.ndarray:
    X = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.mean(numeric_only=True)).to_numpy(dtype=float)


def dsi(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    numeric_cols: Optional[Iterable[str]] = None,
    n_components: int = 5,
    random_state: int = 0,
) -> dict:
    """Compute DSI.

    Args:
        real_df: Real tabular data.
        syn_df: Synthetic tabular data with the same numeric columns.
        numeric_cols: Columns to include. If None, all numeric columns common
            to both dataframes are used.
        n_components: Number of GMM mixture components.
        random_state: GMM fit seed.

    Returns:
        Dict with:
          - "dsi": overall score in (-inf, 1]
          - "n_components": components used
          - "columns": columns used to fit the GMM
          - "real_avg_loglik": average per-row log-likelihood of real data
    """
    if numeric_cols is None:
        numeric_cols = [
            c for c in real_df.columns
            if c in syn_df.columns
            and pd.api.types.is_numeric_dtype(real_df[c])
            and pd.api.types.is_numeric_dtype(syn_df[c])
        ]
    numeric_cols = list(numeric_cols)
    if not numeric_cols:
        return {
            "dsi": float("nan"),
            "n_components": n_components,
            "columns": [],
            "real_avg_loglik": float("nan"),
            "note": "no numeric columns",
        }

    X_real = _numeric_matrix(real_df, numeric_cols)
    X_syn = _numeric_matrix(syn_df, numeric_cols)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
    ).fit(X_real)

    real_per_row = gmm.score_samples(X_real)
    syn_per_row = gmm.score_samples(X_syn)

    ref = float(real_per_row.mean())
    denom = abs(ref) if abs(ref) > 1e-8 else 1e-8
    per_row = 1.0 - np.abs(syn_per_row - ref) / denom

    return {
        "dsi": float(per_row.mean()),
        "n_components": n_components,
        "columns": numeric_cols,
        "real_avg_loglik": ref,
    }
