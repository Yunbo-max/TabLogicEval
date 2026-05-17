"""
Structured prompt for Column Relationship Knowledge Graph (CR-KG) construction.

Replaces the legacy 3-prompt approach with a single unified prompt that asks
the LLM to output a typed knowledge graph over table columns.
"""

from typing import Dict


def get_graph_construction_prompt(column_descriptions: str) -> Dict[str, str]:
    """Return system and user prompts for CR-KG construction.

    The LLM is asked to produce a JSON object with:
      - nodes: list of column names
      - edges: list of typed, directed edges with confidence and rule

    Supported edge types:
      hierarchical  -- functional dependency / granularity levels
      mathematical  -- formula-based derivation
      temporal      -- time-ordering constraint
      semantic      -- domain-specific conditional constraint

    Args:
        column_descriptions: Human-readable descriptions of every column in
            the target dataset (the same string used in the legacy pipeline).

    Returns:
        A dict with keys ``"system"`` and ``"user"`` containing the prompt
        strings.
    """
    system_prompt = (
        "You are a data-analysis expert specializing in relational data modeling. "
        "Your task is to construct a Column Relationship Knowledge Graph (CR-KG) "
        "from column metadata. You must output ONLY valid JSON -- no markdown "
        "fences, no commentary, no extra text."
    )

    user_prompt = f"""Given the following column descriptions for a tabular dataset, construct a Column Relationship Knowledge Graph.

=== Column Descriptions ===
{column_descriptions}

=== Instructions ===
1. Create a node for EVERY column listed above.
2. Identify directed edges between columns. Each edge MUST have:
   - "source": the independent / parent / earlier column name (exact spelling)
   - "target": the dependent / child / later column name (exact spelling)
   - "type": one of "hierarchical", "mathematical", "temporal", "semantic"
   - "confidence": a float between 0.0 and 1.0 representing your certainty
   - "rule": a short natural-language or formula description of the relationship

3. Edge type definitions:
   * hierarchical -- source functionally determines target (same concept at
     different granularity, e.g. City -> State -> Country).  Direction goes
     from MOST SPECIFIC to LEAST SPECIFIC.
   * mathematical -- target is computed from source (and possibly other
     columns) via a deterministic formula.  Include the formula in "rule".
   * temporal -- source datetime precedes target datetime.  Direction goes
     from EARLIER to LATER.
   * semantic -- a domain-specific conditional constraint (e.g. "IF
     ship_mode == 'Air' THEN weight < 100").  Source is the condition
     column, target is the constrained column.

4. Be CONSERVATIVE.  Only propose edges you are highly confident about.
   Do NOT invent relationships that are merely plausible correlations.

5. Each column may appear in multiple edges, but do NOT create duplicate
   edges (same source, target, type).

=== Required Output Format (strict JSON) ===
{{
  "nodes": ["col1", "col2", "..."],
  "edges": [
    {{
      "source": "col_a",
      "target": "col_b",
      "type": "hierarchical",
      "confidence": 0.95,
      "rule": "col_a determines col_b"
    }},
    {{
      "source": "price",
      "target": "total",
      "type": "mathematical",
      "confidence": 1.0,
      "rule": "total = price * quantity"
    }}
  ]
}}

Output ONLY the JSON object. No other text."""

    return {"system": system_prompt, "user": user_prompt}
