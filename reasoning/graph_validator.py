"""
Data-driven validation for CR-KG edges.

Each candidate edge proposed by the LLM ensemble is checked against the actual
data.  Edges whose validation score falls below a threshold are pruned,
producing a validated graph with hallucination statistics.
"""

import re
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-type validators
# ---------------------------------------------------------------------------

def validate_hierarchical(df: pd.DataFrame, source_col: str, target_col: str) -> float:
    """Check functional dependency: does *source* uniquely determine *target*?

    Groups by ``source_col`` and checks whether ``target_col`` has exactly one
    unique value per group.

    Returns:
        Consistency rate in [0, 1].  1.0 means perfect functional dependency.
    """
    if source_col not in df.columns or target_col not in df.columns:
        return 0.0

    try:
        grouped = df.groupby(source_col)[target_col].nunique()
        consistency = (grouped == 1).mean()
        return float(consistency)
    except Exception:
        traceback.print_exc()
        return 0.0


def validate_mathematical(
    df: pd.DataFrame,
    source_cols: List[str],
    target_col: str,
    formula: str,
) -> float:
    """Check whether a mathematical formula holds on the actual data.

    The *formula* should be an expression like ``"total = price * quantity"``.
    We evaluate the right-hand side using column values and compare to the
    left-hand side (``target_col``).

    Args:
        df: The dataset.
        source_cols: Columns referenced on the RHS of the formula.
        target_col: The derived column (LHS).
        formula: A string like ``"target = expr"`` or just ``"expr"``.

    Returns:
        Accuracy rate in [0, 1].  1.0 means the formula matches perfectly.
    """
    if target_col not in df.columns:
        return 0.0
    for col in source_cols:
        if col not in df.columns:
            return 0.0

    try:
        # Parse formula: take the RHS of "lhs = rhs" if present
        if "=" in formula:
            parts = formula.split("=", 1)
            expr = parts[1].strip()
        else:
            expr = formula.strip()

        # Build a safe namespace with column Series
        namespace = {col: df[col].astype(float) for col in source_cols}
        # Also allow numpy functions
        namespace["np"] = np
        namespace["abs"] = np.abs

        computed = eval(expr, {"__builtins__": {}}, namespace)  # noqa: S307
        actual = df[target_col].astype(float)

        # Relative error with epsilon to avoid division by zero
        denom = np.abs(actual) + 1e-8
        rel_error = (np.abs(actual - computed) / denom).mean()
        accuracy = max(0.0, 1.0 - rel_error)
        return float(accuracy)

    except Exception as exc:
        print(f"[validator] mathematical validation failed for '{formula}': {exc}")
        return 0.0


def validate_temporal(df: pd.DataFrame, earlier_col: str, later_col: str) -> float:
    """Check temporal ordering: ``earlier_col <= later_col``.

    Returns:
        Satisfaction rate in [0, 1].  1.0 means no violations.
    """
    if earlier_col not in df.columns or later_col not in df.columns:
        return 0.0

    try:
        earlier = pd.to_datetime(df[earlier_col], errors="coerce")
        later = pd.to_datetime(df[later_col], errors="coerce")

        # Drop rows where either is NaT
        valid = earlier.notna() & later.notna()
        if valid.sum() == 0:
            return 0.0

        violations = (later[valid] < earlier[valid]).sum()
        satisfaction = 1.0 - violations / valid.sum()
        return float(satisfaction)

    except Exception:
        traceback.print_exc()
        return 0.0


def validate_semantic(df: pd.DataFrame, rule: str) -> float:
    """Check a conditional domain constraint expressed as an IF-THEN rule.

    Supported rule patterns (case-insensitive):

    * ``"IF col == 'val' THEN other_col < 100"``
    * ``"IF col != 'val' THEN other_col >= 50"``
    * ``"col IN ('a', 'b') IMPLIES other_col > 0"``

    Falls back to returning 0.5 (unknown) if the rule cannot be parsed.

    Returns:
        Satisfaction rate in [0, 1].
    """
    if not rule:
        return 0.5

    rule_lower = rule.lower().strip()

    try:
        # Try to match "IF <condition> THEN <constraint>"
        match = re.match(
            r"if\s+(.+?)\s+then\s+(.+)",
            rule_lower,
            re.IGNORECASE,
        )
        if not match:
            # Try "IMPLIES" variant
            match = re.match(
                r"(.+?)\s+implies\s+(.+)",
                rule_lower,
                re.IGNORECASE,
            )
        if not match:
            return 0.5  # Cannot parse

        condition_str = match.group(1).strip()
        constraint_str = match.group(2).strip()

        # Evaluate condition and constraint using pandas query
        # Replace column names with backtick-quoted versions for query()
        try:
            condition_mask = df.eval(condition_str)
        except Exception:
            return 0.5

        subset = df[condition_mask]
        if len(subset) == 0:
            return 1.0  # vacuously true

        try:
            constraint_mask = subset.eval(constraint_str)
            satisfaction = constraint_mask.mean()
            return float(satisfaction)
        except Exception:
            return 0.5

    except Exception:
        traceback.print_exc()
        return 0.5


# ---------------------------------------------------------------------------
# Helpers for parsing edge metadata
# ---------------------------------------------------------------------------

def _extract_source_cols_from_rule(rule: str, all_columns: List[str]) -> List[str]:
    """Heuristically extract column names referenced in a formula rule."""
    found = []
    for col in sorted(all_columns, key=len, reverse=True):
        if col in rule:
            found.append(col)
    return found


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_graph(
    df: pd.DataFrame,
    candidate_graph: dict,
    threshold: float = 0.90,
) -> dict:
    """Validate every edge in a candidate CR-KG against actual data.

    Args:
        df: The real dataset.
        candidate_graph: Dict with ``"nodes"`` and ``"edges"`` lists.
        threshold: Minimum validation score to keep an edge.

    Returns:
        A dict with:
          - ``nodes``: same as input
          - ``edges``: edges that survived validation, each augmented with
            ``"validation_score"``
          - ``removed_edges``: edges that failed validation
          - ``stats``: hallucination / validation summary
    """
    edges = candidate_graph.get("edges", [])
    nodes = candidate_graph.get("nodes", [])

    all_columns = list(df.columns)

    validated_edges: List[dict] = []
    removed_edges: List[dict] = []

    for edge in edges:
        edge_type = edge.get("type", "").lower()
        source = edge.get("source", "")
        target = edge.get("target", "")
        rule = edge.get("rule", "")

        score = 0.0

        if edge_type == "hierarchical":
            score = validate_hierarchical(df, source, target)

        elif edge_type == "mathematical":
            source_cols = _extract_source_cols_from_rule(rule, all_columns)
            # Remove target from source cols if accidentally included
            source_cols = [c for c in source_cols if c != target]
            if not source_cols:
                source_cols = [source]
            score = validate_mathematical(df, source_cols, target, rule)

        elif edge_type == "temporal":
            score = validate_temporal(df, source, target)

        elif edge_type == "semantic":
            score = validate_semantic(df, rule)

        else:
            print(f"[validator] Unknown edge type '{edge_type}' -- skipping")
            score = 0.0

        edge_copy = dict(edge)
        edge_copy["validation_score"] = round(score, 4)

        if score >= threshold:
            validated_edges.append(edge_copy)
        else:
            removed_edges.append(edge_copy)
            print(
                f"[validator] PRUNED edge ({source} -> {target}, "
                f"type={edge_type}, score={score:.3f} < {threshold})"
            )

    total = len(edges)
    kept = len(validated_edges)
    removed = len(removed_edges)
    hallucination_rate = removed / total if total > 0 else 0.0

    stats = {
        "total_candidate_edges": total,
        "edges_validated": kept,
        "edges_pruned": removed,
        "hallucination_rate": round(hallucination_rate, 4),
        "threshold": threshold,
    }

    print(f"\n[validator] === Validation Summary ===")
    print(f"  Candidate edges : {total}")
    print(f"  Validated (kept) : {kept}")
    print(f"  Pruned (removed) : {removed}")
    print(f"  Hallucination rate: {hallucination_rate:.1%}")

    return {
        "nodes": nodes,
        "edges": validated_edges,
        "removed_edges": removed_edges,
        "stats": stats,
    }
