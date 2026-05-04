#!/usr/bin/env python3
"""Summarize external-ablation expansion progress toward 20-seed treatment targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from build_external_ablation_eval_datasets import collect_expanded_full_rows


REPO = Path(__file__).resolve().parents[1]
BENCHES = ("math500", "aime2025", "gsm8k")
DEFAULT_OUT = REPO / "data/analysis/external_expansion_progress"
DEFAULT_TARGET = 250


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_statement(text: str) -> str:
    return hashlib.sha256(" ".join(str(text or "").split()).encode("utf-8")).hexdigest()


def _read_jsonl_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _published_treatment_count(bench: str) -> int:
    files = sorted((REPO / "data/eval/external").glob(f"{bench}_treatment_*.jsonl"))
    if not files:
        return 0
    return max(_read_jsonl_count(path) for path in files)


def _manifest_seed_sets() -> tuple[Dict[str, set[str]], Dict[str, set[str]]]:
    manifest = yaml.safe_load((REPO / "data/seed/external_ablation/combo_manifest_ablation.yaml").read_text())
    new_ids: Dict[str, set[str]] = {}
    existing_ids: Dict[str, set[str]] = {}
    for bench, entry in manifest.items():
        if bench not in BENCHES:
            continue
        new_ids[bench] = set(entry["new_all"]["seed_ids"])
        existing_ids[bench] = set(entry["existing_all"]["seed_ids"])
    return existing_ids, new_ids


def _infer_benchmark(seed_ids: Iterable[str], existing_ids: Dict[str, set[str]], new_ids: Dict[str, set[str]]) -> str:
    seeds = set(seed_ids or [])
    for bench in BENCHES:
        if seeds and seeds <= existing_ids[bench] | new_ids[bench]:
            return bench
    return ""


def _seed_split(bench: str, seed_ids: Iterable[str], existing_ids: Dict[str, set[str]], new_ids: Dict[str, set[str]]) -> str:
    seeds = set(seed_ids or [])
    if seeds and seeds <= new_ids[bench]:
        return "new_diagnostic"
    if seeds and seeds <= existing_ids[bench]:
        return "existing_matched"
    return "mixed"


def _iter_ablation_runs() -> Iterable[Dict[str, Any]]:
    existing_ids, new_ids = _manifest_seed_sets()
    for run_dir in sorted((REPO / "data/runs").iterdir()):
        result_path = run_dir / "full_run_result.json"
        if not result_path.exists():
            continue
        try:
            result = _read_json(result_path)
        except Exception:
            continue
        options = result.get("session_options") or {}
        if "external_ablation" not in str(options.get("seed_spec") or ""):
            continue
        seed_ids = list(result.get("seed_ids") or [])
        bench = _infer_benchmark(seed_ids, existing_ids, new_ids)
        if not bench:
            continue
        condition = str((result.get("parameters") or {}).get("ablation_condition") or options.get("ablation_condition") or "full")
        records: List[Dict[str, Any]] = []
        vp_dir = run_dir / "validated_problems"
        for path in sorted(vp_dir.glob("*.json")) if vp_dir.is_dir() else []:
            try:
                payload = _read_json(path)
            except Exception:
                continue
            problem = payload.get("problem") or {}
            records.append(
                {
                    "id": problem.get("id") or path.stem,
                    "statement_hash": _norm_statement(problem.get("statement") or ""),
                    "path": str(path.relative_to(REPO)),
                }
            )
        yield {
            "benchmark": bench,
            "seed_split": _seed_split(bench, seed_ids, existing_ids, new_ids),
            "condition": condition,
            "run_dir": str(run_dir.relative_to(REPO)),
            "seed_ids": ",".join(seed_ids),
            "max_generations": options.get("max_generations"),
            "target_problem_count": options.get("target_problem_count"),
            "raw_count": len(records),
            "statement_unique_count": len({record["statement_hash"] for record in records}),
            "id_unique_count": len({record["id"] for record in records}),
            "records": records,
        }


def summarize(target: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    runs = list(_iter_ablation_runs())
    rows: List[Dict[str, Any]] = []
    for run in runs:
        flat = {key: value for key, value in run.items() if key != "records"}
        rows.append(flat)

    summary_rows: List[Dict[str, Any]] = []
    for bench in BENCHES:
        expanded_rows, expanded_summary, _ = collect_expanded_full_rows(bench)
        provenance_counts = expanded_summary.get("provenance_counts", {})
        published = int(provenance_counts.get("published_existing", _published_treatment_count(bench)))
        recovered = int(provenance_counts.get("partial_recovered", 0))
        new_full_runs = [run for run in runs if run["benchmark"] == bench and run["seed_split"] == "new_diagnostic" and run["condition"] == "full"]
        new_full_hashes = {record["statement_hash"] for run in new_full_runs for record in run["records"]}
        new_all_hashes = {
            record["statement_hash"]
            for run in runs
            if run["benchmark"] == bench and run["seed_split"] == "new_diagnostic"
            for record in run["records"]
        }
        current_full_total = len(expanded_rows)
        summary_rows.append(
            {
                "benchmark": bench,
                "target_statement_unique_treatments": target,
                "published_existing_treatment_count": published,
                "new_full_statement_unique_count": len(new_full_hashes),
                "recovered_partial_kept_count": recovered,
                "new_all_conditions_statement_unique_count": len(new_all_hashes),
                "current_full_total": current_full_total,
                "remaining_to_target": max(0, target - current_full_total),
                "remaining_to_200": max(0, 200 - current_full_total),
                "remaining_to_250": max(0, 250 - current_full_total),
                "new_full_run_count": len(new_full_runs),
                "quality_excluded_count": expanded_summary.get("quality_excluded_count", 0),
                "dedup_excluded_count": expanded_summary.get("dedup_excluded_count", 0),
                "expanded_exact_250_ready": current_full_total >= 250,
            }
        )
    return rows, summary_rows


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
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-target", action="store_true")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    run_rows, summary_rows = summarize(args.target)
    _write_csv(out_dir / "external_expansion_runs.csv", run_rows)
    _write_csv(out_dir / "external_expansion_summary.csv", summary_rows)
    payload = {"target": args.target, "summary": summary_rows}
    (out_dir / "external_expansion_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.require_target:
        underfilled = [row for row in summary_rows if int(row["current_full_total"]) < args.target]
        if underfilled:
            details = ", ".join(f"{row['benchmark']}={row['current_full_total']}/{args.target}" for row in underfilled)
            raise SystemExit(f"expanded treatment target not met: {details}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
