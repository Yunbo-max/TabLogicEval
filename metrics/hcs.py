"""
Hierarchical Consistency Score (HCS).

For each hierarchy group G_k (e.g., {city, state, country}), HCS measures the
fraction of synthetic rows whose tuple of values over G_k appears in the set
C_{k,j} of valid tuples observed in the real data. A valid tuple is one that
the real dataset ever produced for that group.

    HCS = (1 / (M * N)) * sum_{k,j} 1[(x_{i,j})_{i in G_k} in C_k]

where M is the number of synthetic rows, N is the number of hierarchy groups,
and C_k is the set of distinct value-tuples observed over G_k in the real data.
"""

from typing import Iterable, List, Sequence

import pandas as pd


def _row_tuples(df: pd.DataFrame, group: Sequence[str]) -> pd.Series:
    cols = [c for c in group if c in df.columns]
    if len(cols) != len(group):
        missing = set(group) - set(cols)
        raise KeyError(f"Columns missing from dataframe: {sorted(missing)}")
    return df[cols].astype(str).agg("||".join, axis=1)


def hcs(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    hierarchy_groups: Iterable[Sequence[str]],
) -> dict:
    """Compute HCS.

    Args:
        real_df: Real tabular data.
        syn_df: Synthetic tabular data with the same columns.
        hierarchy_groups: Iterable of column-name groups. Each group is a list
            of columns that participate in a single hierarchy chain or
            consistency group, e.g. ["city", "state", "country"].

    Returns:
        Dict with:
          - "hcs": overall score in [0, 1]
          - "per_group": list of per-group scores (same order as input)
          - "groups": echoed group definitions
    """
    groups: List[Sequence[str]] = [tuple(g) for g in hierarchy_groups]
    if not groups:
        return {"hcs": float("nan"), "per_group": [], "groups": []}

    per_group_scores: List[float] = []
    n_rows_syn = len(syn_df)

    for g in groups:
        valid = set(_row_tuples(real_df, g).unique())
        syn_tuples = _row_tuples(syn_df, g)
        hits = syn_tuples.isin(valid).sum()
        per_group_scores.append(float(hits / n_rows_syn) if n_rows_syn else 0.0)

    overall = sum(per_group_scores) / len(per_group_scores)
    return {
        "hcs": float(overall),
        "per_group": per_group_scores,
        "groups": [list(g) for g in groups],
    }
