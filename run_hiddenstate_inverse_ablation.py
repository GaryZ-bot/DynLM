"""
Ablation runner for the hidden-state inverse task.

This script does not change the simulator. It repeatedly runs
wm_hiddenstate_benchmark_v2.py with different inverse-prompt settings so that a
paper can analyze whether hidden-state inference failures come from:

1. observation gap length,
2. masking of recent resistance observations, or
3. whether the approximate equations are revealed in the prompt.

Typical use:
    python run_hiddenstate_inverse_ablation.py

After running GPT on each generated output directory, re-run:
    python run_hiddenstate_inverse_ablation.py --evaluate_existing_llm

The generated folders are placed under wm_hiddenstate_ablation_outputs/ by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_MAIN_SCRIPT = "wm_hiddenstate_benchmark_v2.py"
DEFAULT_ROOT = "wm_hiddenstate_ablation_outputs"


ABLATION_CONFIGS: List[Dict[str, object]] = [
    {
        "name": "A0_gap0_no_mask_noeq",
        "inverse_gap": 0,
        "inverse_mask_mode": "no_mask",
        "inverse_visible_interval": 1,
        "reveal_equations": False,
        "description": "Full current observation. Checks whether fatigue can be estimated when no recent resistance is hidden.",
    },
    {
        "name": "A1_gap3_partial_noeq",
        "inverse_gap": 3,
        "inverse_mask_mode": "partial_resistance",
        "inverse_visible_interval": 2,
        "reveal_equations": False,
        "description": "Short observation gap with sparse intermediate anchors.",
    },
    {
        "name": "A2_gap5_partial_noeq_main",
        "inverse_gap": 5,
        "inverse_mask_mode": "partial_resistance",
        "inverse_visible_interval": 3,
        "reveal_equations": False,
        "description": "Recommended main hidden-state condition: final resistance masked, sparse recent anchors visible.",
    },
    {
        "name": "A3_gap8_all_masked_noeq",
        "inverse_gap": 8,
        "inverse_mask_mode": "all_masked",
        "inverse_visible_interval": 3,
        "reveal_equations": False,
        "description": "Hard hidden-state condition: recent resistance fully hidden.",
    },
    {
        "name": "A4_gap5_partial_reveal_eq",
        "inverse_gap": 5,
        "inverse_mask_mode": "partial_resistance",
        "inverse_visible_interval": 3,
        "reveal_equations": True,
        "description": "Equation-reveal ablation for the recommended main condition.",
    },
]


def run_one_config(main_script: Path, output_dir: Path, cfg: Dict[str, object], args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(main_script),
        "--seed",
        str(args.seed),
        "--n_cases_per_difficulty",
        str(args.n_cases_per_difficulty),
        "--history_length",
        str(args.history_length),
        "--rollout_horizon",
        str(args.rollout_horizon),
        "--inverse_gap",
        str(cfg["inverse_gap"]),
        "--inverse_mask_mode",
        str(cfg["inverse_mask_mode"]),
        "--inverse_visible_interval",
        str(cfg["inverse_visible_interval"]),
        "--output_dir",
        str(output_dir),
    ]
    if bool(cfg["reveal_equations"]):
        cmd.append("--reveal_equations")

    llm_file = output_dir / "all_llm_outputs.jsonl"
    if args.evaluate_existing_llm and llm_file.exists():
        cmd.extend(["--llm_outputs", str(llm_file)])

    print("\nRunning", cfg["name"])
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def collect_ablation_tables(root_dir: Path, configs: List[Dict[str, object]]) -> None:
    rows = []
    wms_rows = []
    summary_rows = []

    for cfg in configs:
        out = root_dir / str(cfg["name"])
        summary_path = out / "summary_results.csv"
        wms_path = out / "world_model_scores.csv"
        if not summary_path.exists() or not wms_path.exists():
            continue

        summary = pd.read_csv(summary_path)
        wms = pd.read_csv(wms_path)
        summary["ablation_name"] = cfg["name"]
        wms["ablation_name"] = cfg["name"]
        for k in ["inverse_gap", "inverse_mask_mode", "inverse_visible_interval", "reveal_equations", "description"]:
            summary[k] = cfg[k]
            wms[k] = cfg[k]
        summary_rows.append(summary)
        wms_rows.append(wms)

        inverse_summary = summary[summary["task_type"] == "inverse"]
        for _, r in inverse_summary.iterrows():
            rows.append({
                "ablation_name": cfg["name"],
                "difficulty": r["difficulty"],
                "model": r["model"],
                "inverse_score": r["score"],
                "inverse_normalized_error": r["normalized_error"],
                "inverse_gap": cfg["inverse_gap"],
                "inverse_mask_mode": cfg["inverse_mask_mode"],
                "reveal_equations": cfg["reveal_equations"],
            })

    pd.DataFrame(rows).to_csv(root_dir / "inverse_ablation_summary.csv", index=False)
    if summary_rows:
        pd.concat(summary_rows, ignore_index=True).to_csv(root_dir / "all_ablation_task_summaries.csv", index=False)
    if wms_rows:
        pd.concat(wms_rows, ignore_index=True).to_csv(root_dir / "all_ablation_world_model_scores.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inverse hidden-state ablations")
    parser.add_argument("--main_script", type=str, default=DEFAULT_MAIN_SCRIPT)
    parser.add_argument("--root_dir", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_cases_per_difficulty", type=int, default=18)
    parser.add_argument("--history_length", type=int, default=12)
    parser.add_argument("--rollout_horizon", type=int, default=30)
    parser.add_argument("--evaluate_existing_llm", action="store_true", help="Use all_llm_outputs.jsonl in each ablation folder if present")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    main_script = Path(args.main_script)
    if not main_script.exists():
        # Also support running from a directory where this script and the main script are together.
        candidate = Path(__file__).resolve().parent / args.main_script
        if candidate.exists():
            main_script = candidate
        else:
            raise FileNotFoundError(f"Cannot find main script: {args.main_script}")

    manifest = []
    for cfg in ABLATION_CONFIGS:
        output_dir = root / str(cfg["name"])
        output_dir.mkdir(parents=True, exist_ok=True)
        run_one_config(main_script, output_dir, cfg, args)
        manifest.append({"output_dir": str(output_dir), **cfg})

    pd.DataFrame(manifest).to_csv(root / "ablation_manifest.csv", index=False)
    collect_ablation_tables(root, ABLATION_CONFIGS)
    print("\nAblation outputs written to:", root.resolve())
    print("Key table:", root / "inverse_ablation_summary.csv")


if __name__ == "__main__":
    main()
