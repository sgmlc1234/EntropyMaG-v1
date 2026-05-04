#!/usr/bin/env python3
"""Plan top-up commands for expanded no_near_copy/no_solvability ablation evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from pathlib import Path
from typing import Dict, List

from summarize_external_expansion_progress import _iter_ablation_runs


REPO = Path(__file__).resolve().parents[1]
BENCHES = ("math500", "aime2025", "gsm8k")
CONDITIONS = ("no_near_copy", "no_solvability")
COMBOS = ("existing_a", "existing_b", "new_a", "new_b", "new_c", "new_d")
DEFAULT_OUT = REPO / "data/analysis/external_expansion_progress/ablation_condition_topup_plan.csv"


def _quote(value: object) -> str:
    return shlex.quote(str(value))


def current_counts() -> Dict[tuple[str, str], int]:
    counts: Dict[tuple[str, str], set[str]] = {}
    for run in _iter_ablation_runs():
        condition = str(run.get("condition") or "")
        bench = str(run.get("benchmark") or "")
        if bench not in BENCHES or condition not in CONDITIONS:
            continue
        key = (bench, condition)
        counts.setdefault(key, set())
        for record in run.get("records", []):
            statement_hash = record.get("statement_hash")
            if statement_hash:
                counts[key].add(statement_hash)
    return {key: len(value) for key, value in counts.items()}


def build_plan(target: int, cushion: int, max_generations: int) -> List[Dict[str, object]]:
    counts = current_counts()
    rows: List[Dict[str, object]] = []
    for bench in BENCHES:
        for condition in CONDITIONS:
            current = counts.get((bench, condition), 0)
            remaining = max(0, target - current)
            planned_total = remaining + (cushion if remaining else 0)
            per_combo = math.ceil(planned_total / len(COMBOS)) if planned_total else 0
            for combo in COMBOS:
                if not per_combo:
                    continue
                cmd = [
                    "tools/launch_ablation_run.sh",
                    bench,
                    combo,
                    condition,
                    str(max_generations),
                    str(per_combo),
                ]
                rows.append(
                    {
                        "benchmark": bench,
                        "condition": condition,
                        "combo": combo,
                        "target_statement_unique_candidates": target,
                        "current_statement_unique_candidates": current,
                        "remaining_to_target": remaining,
                        "planned_total_with_cushion": planned_total,
                        "target_problem_count_per_combo": per_combo,
                        "max_generations": max_generations,
                        "command": " ".join(_quote(part) for part in cmd),
                    }
                )
    return rows


def build_count_rows(target: int) -> List[Dict[str, object]]:
    counts = current_counts()
    rows: List[Dict[str, object]] = []
    for bench in BENCHES:
        for condition in CONDITIONS:
            current = counts.get((bench, condition), 0)
            rows.append(
                {
                    "benchmark": bench,
                    "condition": condition,
                    "target_statement_unique_candidates": target,
                    "current_statement_unique_candidates": current,
                    "remaining_to_target": max(0, target - current),
                    "ready": current >= target,
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--cushion", type=int, default=10)
    parser.add_argument("--max-generations", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = build_plan(args.target, args.cushion, args.max_generations)
    count_rows = build_count_rows(args.target)
    out = args.out if args.out.is_absolute() else REPO / args.out
    write_csv(out, rows)
    counts_out = out.with_name("ablation_condition_counts.csv")
    write_csv(counts_out, count_rows)
    print(
        json.dumps(
            {
                "jobs": len(rows),
                "out": str(out.relative_to(REPO)),
                "counts": str(counts_out.relative_to(REPO)),
                "plan": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
