#!/usr/bin/env python3
"""Create a command matrix for the expanded external-significance evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Dict, List


REPO = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO / "data/eval/external_ablation"
DEFAULT_OUT = REPO / "data/eval/external_ablation/significance_job_matrix.csv"
DEFAULT_MODELS = [
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite-preview",
]
BENCHES = ["math500", "aime2025", "gsm8k"]
ARMS = ["control_20", "new_treatment"]
ALL_ARMS = ["control_20", "new_control_10", "new_treatment"]


def _q(value: object) -> str:
    return shlex.quote(str(value))


def dataset_path(eval_dir: Path, bench: str, arm: str, expanded_target: int) -> Path:
    if arm == "control_20":
        return eval_dir / f"{bench}_control_20.jsonl"
    if arm == "new_control_10":
        return eval_dir / f"{bench}_new_control_10.jsonl"
    return eval_dir / f"{bench}_expanded_full_treatment_{expanded_target}.jsonl"


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    eval_dir = args.eval_dir if args.eval_dir.is_absolute() else REPO / args.eval_dir
    rows: List[Dict[str, object]] = []
    models = args.model or DEFAULT_MODELS
    for bench in args.bench or BENCHES:
        for arm in args.arm or ARMS:
            path = dataset_path(eval_dir, bench, arm, args.expanded_target)
            for model in models:
                out_dir = args.results_dir / bench / arm / model.replace("/", "__")
                cmd = [
                    "python3",
                    "tools/run_direct_no_tool_eval.py",
                    "--dataset-jsonl",
                    str(path.relative_to(REPO) if path.is_relative_to(REPO) else path),
                    "--output-dir",
                    str(out_dir),
                    "--model",
                    model,
                    "--eval-arm",
                    arm,
                    "--repeats",
                    str(args.repeats),
                ]
                if args.resume:
                    cmd.append("--resume")
                rows.append(
                    {
                        "benchmark": bench,
                        "arm": arm,
                        "dataset_jsonl": str(path.relative_to(REPO) if path.is_relative_to(REPO) else path),
                        "dataset_rows": row_count(path),
                        "model": model,
                        "repeats": args.repeats,
                        "output_dir": str(out_dir),
                        "ready": str(path.exists()).lower(),
                        "command": " ".join(_q(part) for part in cmd),
                    }
                )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    ready = [row for row in rows if row.get("ready") == "true"]
    payload = {
        "jobs": len(rows),
        "ready_jobs": len(ready),
        "not_ready_jobs": len(rows) - len(ready),
        "datasets": sorted(
            {
                str(row["dataset_jsonl"]): {
                    "rows": int(row["dataset_rows"]),
                    "ready": row["ready"],
                }
                for row in rows
            }.items()
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--results-dir", type=Path, default=Path("data/eval_results/external_significance"))
    parser.add_argument("--bench", choices=BENCHES, action="append")
    parser.add_argument("--arm", choices=ALL_ARMS, action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expanded-target", type=int, default=250)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    rows = build_rows(args)
    out = args.out if args.out.is_absolute() else REPO / args.out
    write_csv(out, rows)
    summary_path = out.with_name(out.stem + "_summary.json")
    write_summary(summary_path, rows)
    print(json.dumps({"jobs": len(rows), "out": str(out.relative_to(REPO)), "summary": str(summary_path.relative_to(REPO))}, indent=2))


if __name__ == "__main__":
    main()
