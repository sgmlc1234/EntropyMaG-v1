#!/usr/bin/env python3
"""Build generation-wise external treatment slices and optional LLM eval matrix."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_external_ablation_eval_datasets import BENCHES, collect_expanded_full_rows  # noqa: E402


DEFAULT_OUT = REPO / "data/eval/external_ablation/generation_slices"
DEFAULT_MODELS = [
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite-preview",
]


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _stable_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: (row.get("_run_dir", ""), int(row.get("_generation", 0) or 0), row.get("id", "")))


def build_slices(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    summary_rows: List[Dict[str, Any]] = []
    job_rows: List[Dict[str, Any]] = []
    benches = args.bench or list(BENCHES)
    models = args.model or DEFAULT_MODELS
    generations = list(range(args.min_generation, args.max_generation + 1))
    for bench in benches:
        rows, _, _ = collect_expanded_full_rows(bench)
        by_gen: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            provenance = row.get("_provenance_status", "")
            if provenance == "published_existing":
                continue
            generation = int(row.get("_generation", 0) or 0)
            if generation > 0:
                by_gen[generation].append(row)
        for generation in generations:
            gen_rows = _stable_rows(by_gen.get(generation, []))
            pool_path = out_dir / f"{bench}_generation_{generation}_pool_{len(gen_rows)}.jsonl"
            _jsonl_write(pool_path, gen_rows)
            eval_path = pool_path
            eval_rows = len(gen_rows)
            cap_ready = False
            if args.cap_per_generation:
                capped = gen_rows[: args.cap_per_generation]
                cap_path = out_dir / f"{bench}_generation_{generation}_cap_{len(capped)}.jsonl"
                _jsonl_write(cap_path, capped)
                eval_path = cap_path
                eval_rows = len(capped)
                cap_ready = len(capped) >= args.min_rows
            ready = eval_rows >= args.min_rows
            summary_rows.append(
                {
                    "benchmark": bench,
                    "generation": generation,
                    "pool_rows": len(gen_rows),
                    "eval_rows": eval_rows,
                    "min_rows": args.min_rows,
                    "ready": str(ready).lower(),
                    "pool_path": _display(pool_path),
                    "eval_path": _display(eval_path),
                    "cap_ready": str(cap_ready).lower(),
                }
            )
            if args.write_job_matrix:
                for model in models:
                    result_dir = args.results_dir / bench / f"generation_{generation}" / model.replace("/", "__")
                    cmd = [
                        "python3",
                        "tools/run_direct_no_tool_eval.py",
                        "--dataset-jsonl",
                        _display(eval_path),
                        "--output-dir",
                        str(result_dir),
                        "--model",
                        model,
                        "--eval-arm",
                        f"generation_{generation}",
                        "--repeats",
                        str(args.repeats),
                        "--resume",
                    ]
                    job_rows.append(
                        {
                            "benchmark": bench,
                            "generation": generation,
                            "dataset_jsonl": _display(eval_path),
                            "dataset_rows": eval_rows,
                            "model": model,
                            "repeats": args.repeats,
                            "output_dir": str(result_dir),
                            "ready": str(ready).lower(),
                            "command": " ".join(_q(part) for part in cmd),
                        }
                    )
    return summary_rows, job_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
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
    parser.add_argument("--bench", choices=BENCHES, action="append")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-generation", type=int, default=1)
    parser.add_argument("--max-generation", type=int, default=10)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--cap-per-generation", type=int, default=0)
    parser.add_argument("--write-job-matrix", action="store_true")
    parser.add_argument("--job-matrix-out", type=Path, default=DEFAULT_OUT / "generation_eval_job_matrix.csv")
    parser.add_argument("--results-dir", type=Path, default=Path("data/eval_results/external_generation_slices"))
    parser.add_argument("--model", action="append")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    summary_rows, job_rows = build_slices(args)
    summary_csv = out_dir / "generation_slice_summary.csv"
    summary_json = out_dir / "generation_slice_summary.json"
    _write_csv(summary_csv, summary_rows)
    summary_json.write_text(json.dumps({"summary": summary_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.write_job_matrix:
        matrix = args.job_matrix_out if args.job_matrix_out.is_absolute() else REPO / args.job_matrix_out
        _write_csv(matrix, job_rows)
        matrix_summary = {
            "jobs": len(job_rows),
            "ready_jobs": sum(1 for row in job_rows if row.get("ready") == "true"),
            "not_ready_jobs": sum(1 for row in job_rows if row.get("ready") != "true"),
        }
        matrix.with_name(matrix.stem + "_summary.json").write_text(
            json.dumps(matrix_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "summary_csv": _display(summary_csv),
                "summary_json": _display(summary_json),
                "job_matrix_rows": len(job_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
