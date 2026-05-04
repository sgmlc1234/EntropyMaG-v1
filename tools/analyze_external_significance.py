#!/usr/bin/env python3
"""Analyze expanded external-control/treatment model-evaluation significance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Tuple


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO / "data/eval_results/external_significance"
DEFAULT_OUT_DIR = REPO / "data/analysis/external_significance"
COMPARE_ARM = "new_treatment"
CONTROL_ARM = "control_20"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_result_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        if path.name == "summary.json":
            continue
        yield path


def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in _iter_result_files(root):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if payload.get("dry_run"):
            continue
        rel_parts = path.relative_to(root).parts
        benchmark = str(payload.get("benchmark") or (rel_parts[0] if len(rel_parts) > 0 else ""))
        arm = str(payload.get("arm") or (rel_parts[1] if len(rel_parts) > 1 else ""))
        model = str(payload.get("model") or (rel_parts[2].replace("__", "/") if len(rel_parts) > 2 else ""))
        rows.append(
            {
                "benchmark": benchmark,
                "arm": arm,
                "model": model,
                "problem_id": str(payload.get("problem_id") or payload.get("entropymath_id") or payload.get("row_index", "")),
                "problem_key": str(payload.get("row_index", payload.get("problem_id", ""))),
                "repeat": int(payload.get("repeat", 0) or 0),
                "correct": bool(payload.get("correct")),
                "answer_extraction_status": payload.get("answer_extraction_status", ""),
                "path": str(path.relative_to(REPO) if path.is_relative_to(REPO) else path),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def z_test_two_prop(success_a: int, n_a: int, success_b: int, n_b: int) -> Dict[str, float]:
    if min(n_a, n_b) == 0:
        return {"z": 0.0, "p_two_sided": 1.0, "p_one_sided_control_gt_treatment": 1.0}
    p_a = success_a / n_a
    p_b = success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se = math.sqrt(max(1e-12, pooled * (1 - pooled) * (1 / n_a + 1 / n_b)))
    z = (p_a - p_b) / se
    normal = NormalDist()
    p_one = 1 - normal.cdf(z)
    p_two = 2 * min(normal.cdf(z), 1 - normal.cdf(z))
    return {"z": z, "p_two_sided": min(1.0, p_two), "p_one_sided_control_gt_treatment": min(1.0, p_one)}


def by_problem(rows: List[Dict[str, Any]]) -> Dict[str, List[bool]]:
    grouped: Dict[str, List[bool]] = defaultdict(list)
    for row in rows:
        grouped[row["problem_key"]].append(bool(row["correct"]))
    return grouped


def problem_values(problem_runs: Dict[str, List[bool]], metric: str) -> List[float]:
    values: List[float] = []
    for runs in problem_runs.values():
        if not runs:
            continue
        if metric == "pass_at_k":
            values.append(1.0 if any(runs) else 0.0)
        else:
            values.append(sum(1 for value in runs if value) / len(runs))
    return values


def bootstrap_diff(control_values: List[float], treatment_values: List[float], seed: int, n_boot: int) -> Dict[str, float]:
    if not control_values or not treatment_values:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_one_sided_control_gt_treatment": 1.0}
    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(n_boot):
        c = [control_values[rng.randrange(len(control_values))] for _ in control_values]
        t = [treatment_values[rng.randrange(len(treatment_values))] for _ in treatment_values]
        diffs.append(sum(c) / len(c) - sum(t) / len(t))
    diffs.sort()
    low = diffs[int(0.025 * (len(diffs) - 1))]
    high = diffs[int(0.975 * (len(diffs) - 1))]
    mean_diff = sum(control_values) / len(control_values) - sum(treatment_values) / len(treatment_values)
    p_one = sum(1 for diff in diffs if diff <= 0) / len(diffs)
    return {"mean_diff": mean_diff, "ci_low": low, "ci_high": high, "p_one_sided_control_gt_treatment": p_one}


def summarize_cell(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    successes = sum(1 for row in rows if row["correct"])
    low, high = wilson(successes, n)
    problem_runs = by_problem(rows)
    acc_values = problem_values(problem_runs, "accuracy")
    pass_values = problem_values(problem_runs, "pass_at_k")
    return {
        "run_count": n,
        "problem_count": len(problem_runs),
        "correct_runs": successes,
        "run_accuracy": successes / n if n else 0.0,
        "run_accuracy_wilson95_low": low,
        "run_accuracy_wilson95_high": high,
        "problem_mean_accuracy": sum(acc_values) / len(acc_values) if acc_values else 0.0,
        "pass_at_repeats": sum(pass_values) / len(pass_values) if pass_values else 0.0,
    }


def analyze(rows: List[Dict[str, Any]], seed: int, n_boot: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["arm"], row["model"])].append(row)

    cell_rows: List[Dict[str, Any]] = []
    for (benchmark, arm, model), group in sorted(grouped.items()):
        cell_rows.append({"benchmark": benchmark, "arm": arm, "model": model, **summarize_cell(group)})

    comparison_rows: List[Dict[str, Any]] = []
    keys = sorted({(benchmark, model) for benchmark, arm, model in grouped if arm in {CONTROL_ARM, COMPARE_ARM}})
    for benchmark, model in keys:
        control = grouped.get((benchmark, CONTROL_ARM, model), [])
        treatment = grouped.get((benchmark, COMPARE_ARM, model), [])
        if not control or not treatment:
            continue
        c_sum = summarize_cell(control)
        t_sum = summarize_cell(treatment)
        z = z_test_two_prop(c_sum["correct_runs"], c_sum["run_count"], t_sum["correct_runs"], t_sum["run_count"])
        c_probs = by_problem(control)
        t_probs = by_problem(treatment)
        acc_boot = bootstrap_diff(problem_values(c_probs, "accuracy"), problem_values(t_probs, "accuracy"), seed, n_boot)
        pass_boot = bootstrap_diff(problem_values(c_probs, "pass_at_k"), problem_values(t_probs, "pass_at_k"), seed + 17, n_boot)
        comparison_rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "control_arm": CONTROL_ARM,
                "treatment_arm": COMPARE_ARM,
                "control_problems": c_sum["problem_count"],
                "treatment_problems": t_sum["problem_count"],
                "control_run_accuracy": c_sum["run_accuracy"],
                "treatment_run_accuracy": t_sum["run_accuracy"],
                "run_accuracy_drop_pp": 100 * (c_sum["run_accuracy"] - t_sum["run_accuracy"]),
                "run_z": z["z"],
                "run_p_one_sided": z["p_one_sided_control_gt_treatment"],
                "problem_accuracy_drop_pp": 100 * acc_boot["mean_diff"],
                "problem_accuracy_boot95_low_pp": 100 * acc_boot["ci_low"],
                "problem_accuracy_boot95_high_pp": 100 * acc_boot["ci_high"],
                "problem_accuracy_boot_p_one_sided": acc_boot["p_one_sided_control_gt_treatment"],
                "pass_at_repeats_drop_pp": 100 * pass_boot["mean_diff"],
                "pass_at_repeats_boot95_low_pp": 100 * pass_boot["ci_low"],
                "pass_at_repeats_boot95_high_pp": 100 * pass_boot["ci_high"],
                "pass_at_repeats_boot_p_one_sided": pass_boot["p_one_sided_control_gt_treatment"],
            }
        )
    return cell_rows, comparison_rows


def write_markdown(path: Path, comparisons: List[Dict[str, Any]]) -> None:
    lines = [
        "# Expanded external significance analysis",
        "",
        "One-sided p-values test whether control accuracy exceeds treatment accuracy. Bootstrap intervals resample dataset row indices, not individual repeats.",
        "",
        "| Benchmark | Model | N ctl/trt | Run drop pp | Problem bootstrap 95% CI pp | p(one-sided) | Pass@repeats drop pp |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {benchmark} | {model} | {control_problems}/{treatment_problems} | {drop:.1f} | [{low:.1f}, {high:.1f}] | {p:.4f} | {pass_drop:.1f} |".format(
                benchmark=row["benchmark"],
                model=row["model"],
                control_problems=row["control_problems"],
                treatment_problems=row["treatment_problems"],
                drop=row["problem_accuracy_drop_pp"],
                low=row["problem_accuracy_boot95_low_pp"],
                high=row["problem_accuracy_boot95_high_pp"],
                p=row["problem_accuracy_boot_p_one_sided"],
                pass_drop=row["pass_at_repeats_drop_pp"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260502)
    args = parser.parse_args()

    root = args.results_root if args.results_root.is_absolute() else REPO / args.results_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    rows = load_rows(root)
    cell_rows, comparison_rows = analyze(rows, args.seed, args.bootstrap)
    write_csv(out_dir / "external_significance_runs.csv", rows)
    write_csv(out_dir / "external_significance_cells.csv", cell_rows)
    write_csv(out_dir / "external_significance_comparisons.csv", comparison_rows)
    write_markdown(out_dir / "external_significance_summary.md", comparison_rows)
    print(json.dumps({"runs": len(rows), "cells": len(cell_rows), "comparisons": len(comparison_rows), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
