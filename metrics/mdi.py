"""
Multivariate Dependency Index (MDI).

For each dependency rule D_{g,j} over a group of columns G_g (mathematical or
temporal), MDI measures the fraction of synthetic rows that satisfy D_{g,j}.

    MDI = (1 / (M * N)) * sum_{g,j} 1[D_{g,j} holds in row j]

A rule is one of:
  * mathematical: target = f(sources), e.g. "sales = quantity * price".
    Tolerance is relative (default 1e-3).
  * temporal:     source_date <= target_date.

Rules are passed as dicts; see Rule type below for the schema.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class Rule:
    """A single dependency rule over the table's columns.

    For ``kind == "mathematical"``:
        target:   name of the derived column (LHS).
        formula:  Python expression in column names, e.g. "quantity * price"
                  or "price * quantity * (1 - discount_rate)". The expression
                  must evaluate using pandas vector operations.

    For ``kind == "temporal"``:
        source:   earlier datetime column.
        target:   later datetime column. Rule satisfied when source <= target.
    """
    kind: str
    target: str
    formula: Optional[str] = None
    source: Optional[str] = None
    rtol: float = 1e-3


def _check_mathematical(df: pd.DataFrame, rule: Rule) -> np.ndarray:
    if rule.formula is None:
        raise ValueError(f"Mathematical rule for {rule.target} missing formula")
    cols = {c: df[c].astype(float) for c in df.columns if c in rule.formula}
    cols["np"] = np
    predicted = eval(rule.formula, {"__builtins__": {}}, cols)  # noqa: S307
    actual = df[rule.target].astype(float).to_numpy()
    predicted = np.asarray(predicted, dtype=float)
    denom = np.maximum(np.abs(actual), 1e-8)
    return np.abs(actual - predicted) / denom <= rule.rtol


def _check_temporal(df: pd.DataFrame, rule: Rule) -> np.ndarray:
    if rule.source is None:
        raise ValueError(f"Temporal rule for {rule.target} missing source")
    earlier = pd.to_datetime(df[rule.source], errors="coerce")
    later = pd.to_datetime(df[rule.target], errors="coerce")
    valid = earlier.notna() & later.notna()
    ok = np.zeros(len(df), dtype=bool)
    ok[valid] = (earlier[valid] <= later[valid]).to_numpy()
    return ok


def mdi(syn_df: pd.DataFrame, rules: Iterable[Rule]) -> dict:
    """Compute MDI over a list of dependency rules.

    Args:
        syn_df: Synthetic tabular data.
        rules: Iterable of Rule. Rules referencing missing columns are skipped
            and recorded in the ``"skipped"`` list of the result.

    Returns:
        Dict with:
          - "mdi": overall score in [0, 1]
          - "per_rule": list of {"rule": ..., "score": ...}
          - "skipped": rules skipped because columns were missing
    """
    rules = list(rules)
    if not rules:
        return {"mdi": float("nan"), "per_rule": [], "skipped": []}

    per_rule: List[dict] = []
    skipped: List[dict] = []
    n = len(syn_df)

    for r in rules:
        required = {r.target}
        if r.kind == "mathematical":
            required.update(c for c in syn_df.columns if c in (r.formula or ""))
        if r.kind == "temporal" and r.source:
            required.add(r.source)
        missing = required - set(syn_df.columns)
        if missing:
            skipped.append({"rule": r.__dict__, "missing": sorted(missing)})
            continue

        try:
            if r.kind == "mathematical":
                ok = _check_mathematical(syn_df, r)
            elif r.kind == "temporal":
                ok = _check_temporal(syn_df, r)
            else:
                skipped.append({"rule": r.__dict__, "error": f"unknown kind: {r.kind}"})
                continue
            score = float(ok.sum()) / n if n else 0.0
            per_rule.append({"rule": r.__dict__, "score": score})
        except Exception as exc:
            skipped.append({"rule": r.__dict__, "error": str(exc)})

    overall = sum(p["score"] for p in per_rule) / len(per_rule) if per_rule else float("nan")
    return {"mdi": float(overall), "per_rule": per_rule, "skipped": skipped}
