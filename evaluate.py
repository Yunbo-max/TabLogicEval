"""
End-to-end TabLogicEval entry point.

    real.csv + syn.csv + column descriptions
        -> identify(real, descriptions)            # LLM-derived G_k, rules
        -> HCS(real, syn, G_k)
        -> MDI(syn, rules)
        -> DSI(real, syn)
        -> JSON report
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from identify import identify
from metrics import dsi, hcs, mdi

load_dotenv()


def _serialise(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj


def evaluate(
    real_path: str,
    syn_path: str,
    descriptions: str,
    models,
    temperatures,
    temp: float,
    max_tok: int,
    validation_threshold: float,
    dsi_components: int,
    out_path: str | None,
) -> dict:
    real = pd.read_csv(real_path)
    syn = pd.read_csv(syn_path)

    common = [c for c in real.columns if c in syn.columns]
    if len(common) != len(real.columns):
        print(f"[evaluate] Restricting to {len(common)} common columns "
              f"(real had {len(real.columns)}, syn had {len(syn.columns)}).")
    real, syn = real[common], syn[common]

    groups, rules, report = identify(
        real,
        descriptions,
        models=models,
        temperatures=temperatures,
        temp=temp,
        max_tok=max_tok,
        validation_threshold=validation_threshold,
    )

    print(f"[evaluate] {len(groups)} hierarchy groups, {len(rules)} dependency rules")
    hcs_result = hcs(real, syn, groups)
    mdi_result = mdi(syn, rules)
    dsi_result = dsi(real, syn, n_components=dsi_components)

    summary = {
        "real": real_path,
        "synthetic": syn_path,
        "scores": {
            "HCS": hcs_result["hcs"],
            "MDI": mdi_result["mdi"],
            "DSI": dsi_result["dsi"],
        },
        "hcs": hcs_result,
        "mdi": _serialise(mdi_result),
        "dsi": dsi_result,
        "identification": report,
    }

    print("\n=== TabLogicEval scores ===")
    for k, v in summary["scores"].items():
        print(f"  {k}: {v:.4f}" if v == v else f"  {k}: nan")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nReport written to {out_path}")

    return summary


def _load_descriptions(path: str) -> str:
    if path and os.path.exists(path):
        return Path(path).read_text()
    return path or ""


def main():
    p = argparse.ArgumentParser(
        description="TabLogicEval: HCS / MDI / DSI evaluation of synthetic tabular data."
    )
    p.add_argument("--real", required=True, help="Path to real CSV.")
    p.add_argument("--syn", required=True, help="Path to synthetic CSV.")
    p.add_argument(
        "--descriptions",
        required=True,
        help="Either a path to a text file with column descriptions, or the "
             "descriptions string itself.",
    )
    p.add_argument("--models", default="deepseek",
                   help="Comma-separated LLM model names for the identification ensemble.")
    p.add_argument("--temperatures", default=None,
                   help="Comma-separated temperatures for same-model voting.")
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--max_tok", type=int, default=2000)
    p.add_argument("--validation_threshold", type=float, default=0.90)
    p.add_argument("--dsi_components", type=int, default=5)
    p.add_argument("--out", default="results/report.json",
                   help="Where to write the JSON report. Use 'none' to skip.")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    temperatures = (
        [float(t) for t in args.temperatures.split(",") if t.strip()]
        if args.temperatures else None
    )
    out_path = None if args.out.lower() == "none" else args.out

    evaluate(
        real_path=args.real,
        syn_path=args.syn,
        descriptions=_load_descriptions(args.descriptions),
        models=models,
        temperatures=temperatures,
        temp=args.temp,
        max_tok=args.max_tok,
        validation_threshold=args.validation_threshold,
        dsi_components=args.dsi_components,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
