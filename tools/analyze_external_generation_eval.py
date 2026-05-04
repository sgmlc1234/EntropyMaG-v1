#!/usr/bin/env python3
"""Analyze direct no-tool evaluation results by external generation slice."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO / "data/eval_results/external_generation_slices"
DEFAULT_OUT_DIR = REPO / "data/analysis/external_generation_eval"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_result_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        if path.name == "summary.json":
            continue
        yield path


def _generation_from_payload(payload: Dict[str, Any], path: Path, root: Path) -> int:
    arm = str(payload.get("arm") or "")
    match = re.search(r"generation_(\d+)", arm)
    if match:
        return int(match.group(1))
    for part in path.relative_to(root).parts:
        match = re.fullmatch(r"generation_(\d+)", part)
        if match:
            return int(match.group(1))
    return 0


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
        rows.append(
            {
                "benchmark": str(payload.get("benchmark") or (rel_parts[0] if len(rel_parts) > 0 else "")),
                "generation": _generation_from_payload(payload, path, root),
                "model": str(payload.get("model") or (rel_parts[2].replace("__", "/") if len(rel_parts) > 2 else "")),
                "problem_id": str(payload.get("problem_id") or payload.get("row_index", "")),
                "repeat": int(payload.get("repeat", 0) or 0),
                "correct": bool(payload.get("correct")),
                "answer_extraction_status": str(payload.get("answer_extraction_status") or ""),
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


def _linear_slope(points: List[Tuple[int, float]]) -> float:
    points = [(x, y) for x, y in points if x > 0]
    if len(points) < 2:
        return 0.0
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denom


def summarize(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], int(row["generation"]), row["model"])].append(row)

    cell_rows: List[Dict[str, Any]] = []
    for (benchmark, generation, model), group in sorted(grouped.items()):
        run_count = len(group)
        correct = sum(1 for row in group if row["correct"])
        problem_count = len({row["problem_id"] for row in group})
        extraction_fail = sum(1 for row in group if row["answer_extraction_status"] != "boxed_found")
        cell_rows.append(
            {
                "benchmark": benchmark,
                "generation": generation,
                "model": model,
                "problem_count": problem_count,
                "run_count": run_count,
                "correct_runs": correct,
                "run_accuracy": correct / run_count if run_count else 0.0,
                "extraction_fail_runs": extraction_fail,
            }
        )

    trend_rows: List[Dict[str, Any]] = []
    by_series: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        by_series[(row["benchmark"], row["model"])].append(row)
    for (benchmark, model), series in sorted(by_series.items()):
        series = sorted(series, key=lambda row: int(row["generation"]))
        points = [(int(row["generation"]), float(row["run_accuracy"])) for row in series]
        first = points[0][1] if points else 0.0
        last = points[-1][1] if points else 0.0
        trend_rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "generations": len(points),
                "first_generation_accuracy": first,
                "last_generation_accuracy": last,
                "last_minus_first_pp": 100 * (last - first),
                "linear_slope_pp_per_generation": 100 * _linear_slope(points),
                "min_accuracy": min((value for _, value in points), default=0.0),
                "max_accuracy": max((value for _, value in points), default=0.0),
            }
        )
    return cell_rows, trend_rows


def write_markdown(path: Path, trend_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# External generation-wise direct evaluation",
        "",
        "Positive slopes mean accuracy increased with generation; negative slopes mean later generations were harder for the model.",
        "",
        "| Benchmark | Model | Gen1 acc | Gen10 acc | Δ pp | Slope pp/gen |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in trend_rows:
        lines.append(
            "| {benchmark} | {model} | {first:.3f} | {last:.3f} | {delta:.1f} | {slope:.2f} |".format(
                benchmark=row["benchmark"],
                model=row["model"],
                first=row["first_generation_accuracy"],
                last=row["last_generation_accuracy"],
                delta=row["last_minus_first_pp"],
                slope=row["linear_slope_pp_per_generation"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    root = args.results_root if args.results_root.is_absolute() else REPO / args.results_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    rows = load_rows(root)
    cell_rows, trend_rows = summarize(rows)
    write_csv(out_dir / "external_generation_runs.csv", rows)
    write_csv(out_dir / "external_generation_cells.csv", cell_rows)
    write_csv(out_dir / "external_generation_trends.csv", trend_rows)
    write_markdown(out_dir / "external_generation_summary.md", trend_rows)
    print(
        json.dumps(
            {
                "runs": len(rows),
                "cells": len(cell_rows),
                "trends": len(trend_rows),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
