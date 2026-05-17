"""
LLM-based identification of hierarchy groups and dependency rules.

Given a real CSV and a free-text column-metadata description, build a
validated Column Relationship Knowledge Graph (CR-KG) using the multi-LLM
ensemble + data-driven validator copied from TabKG, then convert the
validated edges into:

    - hierarchy_groups : list[list[str]]   (input to HCS)
    - rules            : list[Rule]        (input to MDI)

This module is thus the "front end" that wires LLM-inferred logical structure
into the HCS and MDI metrics so the user does not have to specify G_k and
D_{g,j} manually.
"""

from collections import defaultdict
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pandas as pd

from metrics.mdi import Rule
from reasoning.graph_validator import validate_graph
from reasoning.llm_ensemble import call_single_model, run_ensemble


def _candidate_graph(
    column_descriptions: str,
    models: List[str],
    temperatures: Optional[List[float]],
    temp: float,
    max_tok: int,
) -> dict:
    args = SimpleNamespace(temp=temp, max_tok=max_tok)
    if len(models) == 1 and not temperatures:
        g = call_single_model(models[0], column_descriptions, args)
        if g is None:
            return {"nodes": [], "edges": [], "stats": {"n_success": 0}}
        return {
            "nodes": g.get("nodes", []),
            "edges": g.get("edges", []),
            "stats": {"n_models": 1, "n_success": 1},
        }
    return run_ensemble(column_descriptions, models, args, temperatures=temperatures)


def _chains_from_hierarchical_edges(edges: List[dict]) -> List[List[str]]:
    """Collapse hierarchical edges (source -> target = parent) into chains.

    Treats each weakly-connected component of the hierarchical subgraph as a
    single hierarchy group, ordered from most-specific (root) to least-specific
    (leaf) along the edge direction.
    """
    children = defaultdict(set)
    parents = defaultdict(set)
    nodes = set()
    for e in edges:
        if e.get("type", "").lower() != "hierarchical":
            continue
        s, t = e["source"], e["target"]
        children[s].add(t)
        parents[t].add(s)
        nodes.update([s, t])

    visited = set()
    groups: List[List[str]] = []
    for start in nodes:
        if start in visited:
            continue
        component = set()
        frontier = [start]
        while frontier:
            n = frontier.pop()
            if n in component:
                continue
            component.add(n)
            frontier.extend(children[n] | parents[n])
        visited |= component

        order: List[str] = []
        indeg = {n: len(parents[n] & component) for n in component}
        remaining = set(component)
        while remaining:
            roots = sorted(n for n in remaining if indeg[n] == 0)
            if not roots:
                order.extend(sorted(remaining))
                break
            order.extend(roots)
            for r in roots:
                remaining.discard(r)
                for c in children[r]:
                    if c in indeg:
                        indeg[c] -= 1
        groups.append(order)
    return groups


def _rules_from_edges(edges: List[dict]) -> List[Rule]:
    rules: List[Rule] = []
    for e in edges:
        t = e.get("type", "").lower()
        if t == "mathematical":
            formula = e.get("rule", "") or ""
            if "=" in formula:
                formula = formula.split("=", 1)[1].strip()
            if not formula:
                continue
            rules.append(Rule(kind="mathematical", target=e["target"], formula=formula))
        elif t == "temporal":
            rules.append(Rule(kind="temporal", target=e["target"], source=e["source"]))
    return rules


def identify(
    real_df: pd.DataFrame,
    column_descriptions: str,
    models: Optional[List[str]] = None,
    temperatures: Optional[List[float]] = None,
    temp: float = 0.1,
    max_tok: int = 2000,
    validation_threshold: float = 0.90,
) -> Tuple[List[List[str]], List[Rule], dict]:
    """Run LLM ensemble + data validation, then derive HCS groups and MDI rules.

    Returns ``(hierarchy_groups, rules, report)``.
    """
    models = models or ["deepseek"]
    candidate = _candidate_graph(column_descriptions, models, temperatures, temp, max_tok)
    validated = validate_graph(real_df, candidate, threshold=validation_threshold)

    edges = validated["edges"]
    hierarchy_groups = _chains_from_hierarchical_edges(edges)
    rules = _rules_from_edges(edges)

    report = {
        "candidate_stats": candidate.get("stats", {}),
        "validated_stats": validated.get("stats", {}),
        "n_hierarchy_groups": len(hierarchy_groups),
        "n_rules": len(rules),
        "removed_edges": validated.get("removed_edges", []),
    }
    return hierarchy_groups, rules, report
