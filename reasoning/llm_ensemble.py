"""
Multi-LLM ensemble with majority voting for CR-KG construction.

Supports: gpt-4, deepseek, claude, llama (together.ai / local placeholder).
Each model proposes a candidate graph; edges agreed upon by a majority are kept.
"""

import json
import math
import os
import re
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import requests

from .prompts import get_graph_construction_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from an LLM response string."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _edge_key(edge: dict) -> Tuple[str, str, str]:
    """Canonical key for deduplication: (source, target, type) all lowercased."""
    return (
        edge["source"].strip().lower(),
        edge["target"].strip().lower(),
        edge["type"].strip().lower(),
    )


# ---------------------------------------------------------------------------
# Per-model callers
# ---------------------------------------------------------------------------

def _call_openai(prompt: Dict[str, str], args: Any, model_id: str = "gpt-4o") -> Optional[dict]:
    """Call OpenAI GPT-4o (or compatible) via the openai SDK."""
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"[ensemble] OPENAI_API_KEY not set -- skipping {model_id}")
        return None

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        temperature=getattr(args, "temp", 0.1),
        max_tokens=getattr(args, "max_tok", 2000),
    )
    content = response.choices[0].message.content
    return _extract_json(content)


def _call_deepseek(prompt: Dict[str, str], args: Any) -> Optional[dict]:
    """Call DeepSeek chat API via HTTP."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[ensemble] DEEPSEEK_API_KEY not set -- skipping deepseek")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "temperature": getattr(args, "temp", 0.1),
        "max_tokens": getattr(args, "max_tok", 2000),
    }
    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _extract_json(content)


def _call_claude(prompt: Dict[str, str], args: Any) -> Optional[dict]:
    """Call Anthropic Claude via the anthropic SDK."""
    try:
        import anthropic
    except ImportError:
        print("[ensemble] anthropic SDK not installed -- skipping claude")
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[ensemble] ANTHROPIC_API_KEY not set -- skipping claude")
        return None

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=getattr(args, "max_tok", 2000),
        system=prompt["system"],
        messages=[{"role": "user", "content": prompt["user"]}],
        temperature=getattr(args, "temp", 0.1),
    )
    content = message.content[0].text
    return _extract_json(content)


def _call_llama(prompt: Dict[str, str], args: Any) -> Optional[dict]:
    """Call Llama model via Together.ai (OpenAI-compatible endpoint)."""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("[ensemble] TOGETHER_API_KEY not set -- skipping llama")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "temperature": getattr(args, "temp", 0.1),
        "max_tokens": getattr(args, "max_tok", 2000),
    }
    resp = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _extract_json(content)


def _call_siliconflow(model_id: str):
    """Factory for SiliconFlow-hosted models (Qwen, MiniMax, etc.)."""
    def _caller(prompt: Dict[str, str], args: Any) -> Optional[dict]:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            print(f"[ensemble] SILICONFLOW_API_KEY not set -- skipping {model_id}")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            "temperature": getattr(args, "temp", 0.1),
            "max_tokens": getattr(args, "max_tok", 2000),
        }
        resp = requests.post(
            "https://api.siliconflow.cn/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _extract_json(content)
    return _caller


def _call_gpt4o(prompt, args):
    return _call_openai(prompt, args, model_id="gpt-4o")

def _call_gpt4o_mini(prompt, args):
    return _call_openai(prompt, args, model_id="gpt-4o-mini")

def _call_gpt4(prompt, args):
    return _call_openai(prompt, args, model_id="gpt-4")

def _call_gpt41(prompt, args):
    return _call_openai(prompt, args, model_id="gpt-4.1")


_MODEL_DISPATCH = {
    "gpt-4.1": _call_gpt41,
    "gpt-4o": _call_gpt4o,
    "gpt-4o-mini": _call_gpt4o_mini,
    "gpt-4": _call_gpt4,
    "gpt": _call_gpt4o,
    "deepseek": _call_deepseek,
    "claude": _call_claude,
    "llama": _call_llama,
    "qwen": _call_siliconflow("Qwen/Qwen3.5-397B-A17B"),
    "minimax": _call_siliconflow("Pro/MiniMaxAI/MiniMax-M2.5"),
    "glm": _call_siliconflow("Pro/zai-org/GLM-5"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_single_model(model_name: str, column_descriptions: str, args: Any) -> Optional[dict]:
    """Run graph construction on a single LLM.

    Args:
        model_name: One of ``"gpt-4"``, ``"deepseek"``, ``"claude"``, ``"llama"``.
        column_descriptions: Column metadata string.
        args: Namespace with ``temp`` and ``max_tok`` attributes.

    Returns:
        Parsed graph dict (``{"nodes": [...], "edges": [...]}``) or ``None``
        on failure.
    """
    caller = _MODEL_DISPATCH.get(model_name)
    if caller is None:
        print(f"[ensemble] Unknown model: {model_name}")
        return None

    prompt = get_graph_construction_prompt(column_descriptions)
    print(f"[ensemble] Calling {model_name} ...")
    try:
        graph = caller(prompt, args)
        if graph and "nodes" in graph and "edges" in graph:
            print(f"[ensemble] {model_name} returned {len(graph['edges'])} edges")
            return graph
        else:
            print(f"[ensemble] {model_name} returned invalid graph structure")
            return None
    except Exception as exc:
        print(f"[ensemble] {model_name} failed: {exc}")
        traceback.print_exc()
        return None


def run_ensemble(
    column_descriptions: str,
    models: List[str],
    args: Any,
    min_agreement: Optional[int] = None,
    temperatures: Optional[List[float]] = None,
) -> dict:
    """Run CR-KG construction on multiple LLMs and apply majority voting.

    Args:
        column_descriptions: Column metadata string.
        models: List of model identifiers (e.g. ``["gpt-4", "deepseek"]``).
        args: Namespace with ``temp`` and ``max_tok`` attributes.
        min_agreement: Minimum number of models that must agree on an edge
            for it to be kept.  Defaults to ``ceil(len(models) / 2)``.
        temperatures: Optional list of floats, one per entry in *models*.
            When provided and all entries in *models* are the same model,
            each run uses the corresponding temperature instead of
            ``args.temp``.  Results are labelled like ``"deepseek_t0.1"``
            for diagnostics.

    Returns:
        A dict with:
          - ``nodes``: union of all nodes across models
          - ``edges``: list of edges surviving majority vote, each augmented
            with an ``"agreement"`` field (fraction of models that proposed it)
          - ``model_results``: per-model raw graphs (for diagnostics)
          - ``stats``: summary statistics
    """
    if min_agreement is None:
        min_agreement = math.ceil(len(models) / 2)

    # Determine if we are in same-model temperature-sweep mode
    _same_model_sweep = (
        temperatures is not None
        and len(temperatures) == len(models)
        and len(set(models)) == 1
    )

    # Collect per-model results
    model_graphs: Dict[str, Optional[dict]] = {}
    for idx, model_name in enumerate(models):
        if _same_model_sweep:
            temp = temperatures[idx]
            label = f"{model_name}_t{temp}"
            # Temporarily override args.temp for this run
            orig_temp = getattr(args, "temp", 0.1)
            args.temp = temp
            print(f"[ensemble] Temperature sweep: {label}")
            model_graphs[label] = call_single_model(model_name, column_descriptions, args)
            args.temp = orig_temp
        else:
            model_graphs[model_name] = call_single_model(model_name, column_descriptions, args)

    successful = {k: v for k, v in model_graphs.items() if v is not None}
    n_success = len(successful)

    if n_success == 0:
        print("[ensemble] All models failed -- returning empty graph")
        return {
            "nodes": [],
            "edges": [],
            "model_results": model_graphs,
            "stats": {"n_models": len(models), "n_success": 0},
        }

    # Recalculate min_agreement based on successful models
    min_agreement = min(min_agreement, math.ceil(n_success / 2))
    min_agreement = max(min_agreement, 1)

    # Collect all nodes (union)
    all_nodes: set = set()
    for g in successful.values():
        all_nodes.update(g.get("nodes", []))

    # Count edge occurrences by canonical key
    edge_votes: Counter = Counter()
    edge_examples: Dict[Tuple[str, str, str], dict] = {}

    for g in successful.values():
        seen_in_model: set = set()
        for edge in g.get("edges", []):
            key = _edge_key(edge)
            if key not in seen_in_model:
                edge_votes[key] += 1
                seen_in_model.add(key)
            # Keep the edge with the highest confidence as representative
            if key not in edge_examples or edge.get("confidence", 0) > edge_examples[key].get("confidence", 0):
                edge_examples[key] = edge

    # Filter by majority vote
    final_edges: List[dict] = []
    total_proposed = len(edge_votes)
    for key, count in edge_votes.items():
        if count >= min_agreement:
            edge = dict(edge_examples[key])
            edge["agreement"] = round(count / n_success, 2)
            final_edges.append(edge)

    # Sort by agreement then confidence (descending)
    final_edges.sort(
        key=lambda e: (e.get("agreement", 0), e.get("confidence", 0)),
        reverse=True,
    )

    stats = {
        "n_models": len(models),
        "n_success": n_success,
        "min_agreement": min_agreement,
        "total_candidate_edges": total_proposed,
        "edges_after_voting": len(final_edges),
        "edges_removed_by_voting": total_proposed - len(final_edges),
    }

    print(f"\n[ensemble] === Majority Voting Summary ===")
    print(f"  Models queried : {len(models)}")
    print(f"  Models succeeded: {n_success}")
    print(f"  Candidate edges : {total_proposed}")
    print(f"  Edges kept (>={min_agreement} votes): {len(final_edges)}")
    print(f"  Edges removed   : {total_proposed - len(final_edges)}")

    return {
        "nodes": sorted(all_nodes),
        "edges": final_edges,
        "model_results": model_graphs,
        "stats": stats,
    }
