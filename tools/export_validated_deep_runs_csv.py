#!/usr/bin/env python3
"""Export all validated problems from historical *-deep-run directories to CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CSV_COLUMNS = [
    "id",
    "statement",
    "answer",
    "solution",
    "verification_code",
    "operation",
    "difficulty",
    "difficulty_label",
    "generation",
    "source_run",
    "source_file",
    "source_slot",
    "parent_ids",
    "ancestor_ids",
    "statement_sha256",
    "answer_sha256",
]


def _sha256(value: Any) -> str:
    return hashlib.sha256(str(value if value is not None else "").encode("utf-8")).hexdigest()


def _json_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _read_problem(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    problem = payload.get("problem", payload)
    meta = payload.get("meta", {})
    if not isinstance(problem, dict):
        raise ValueError(f"{path} has no problem object")
    if not isinstance(meta, dict):
        meta = {}
    return problem, meta


def _difficulty_label(problem: Dict[str, Any]) -> str:
    label = str(problem.get("difficulty_label") or problem.get("Difficulty_Label") or "").strip().lower()
    if label:
        return label
    difficulty = problem.get("difficulty", problem.get("Difficulty", ""))
    try:
        score = float(difficulty)
    except (TypeError, ValueError):
        return ""
    if score < 5:
        return "easy"
    if score < 7:
        return "medium"
    if score < 9:
        return "hard"
    return "superhard"


def _load_generation_index(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for generation_path in sorted(run_dir.glob("generation_*.json")):
        try:
            payload = json.loads(generation_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for problem in payload:
            if not isinstance(problem, dict):
                continue
            problem_id = problem.get("id") or problem.get("ID")
            if problem_id and problem_id not in index:
                index[str(problem_id)] = problem
    return index


def _merge_generation_fallback(
    problem: Dict[str, Any],
    fallback: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not fallback:
        return problem
    merged = dict(fallback)
    merged.update({key: value for key, value in problem.items() if value not in (None, "", [], {})})
    return merged


def _normalize(path: Path, root: Path, generation_indexes: Dict[Path, Dict[str, Dict[str, Any]]]) -> Dict[str, str]:
    problem, meta = _read_problem(path)
    run_dir = path.parent.parent
    problem_id = str(problem.get("id") or problem.get("ID") or path.stem).strip()
    if not (problem.get("statement") or problem.get("Question")):
        generation_index = generation_indexes.setdefault(run_dir, _load_generation_index(run_dir))
        problem = _merge_generation_fallback(problem, generation_index.get(problem_id))
    statement = str(problem.get("statement") or problem.get("Question") or "").strip()
    answer = str(problem.get("answer", problem.get("Answer", ""))).strip()
    operation = str(problem.get("type") or problem.get("op_type") or "").strip()
    source_run = run_dir.name
    row = {
        "id": str(problem.get("id") or problem.get("ID") or problem_id).strip(),
        "statement": statement,
        "answer": answer,
        "solution": str(problem.get("solution") or problem.get("Solution") or "").strip(),
        "verification_code": str(problem.get("verification_code") or problem.get("code") or "").strip(),
        "operation": operation,
        "difficulty": str(problem.get("difficulty", problem.get("Difficulty", ""))).strip(),
        "difficulty_label": _difficulty_label(problem),
        "generation": str(meta.get("generation_count") if meta.get("generation_count") is not None else ""),
        "source_run": source_run,
        "source_file": str(path.relative_to(root)),
        "source_slot": str(meta.get("slot") if meta.get("slot") is not None else problem.get("_slot", "")),
        "parent_ids": _json_text(problem.get("parent_ids", [])),
        "ancestor_ids": _json_text(problem.get("ancestor_ids", [])),
        "statement_sha256": _sha256(statement),
        "answer_sha256": _sha256(answer),
    }
    return row


def _iter_validated_paths(runs_dir: Path) -> Iterable[Path]:
    yield from sorted(runs_dir.glob("*-deep-run/validated_problems/*.json"))


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_path(path: Path, root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return f"<outside-manifest-root>/{resolved.name}"


def _dedupe(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    seen = set()
    kept = []
    dropped = []
    for row in rows:
        key = (row["id"], row["statement_sha256"])
        if key in seen:
            dropped.append(row)
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


def export(args: argparse.Namespace) -> Dict[str, Any]:
    runs_dir = args.runs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_root = args.manifest_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []
    generation_indexes: Dict[Path, Dict[str, Dict[str, Any]]] = {}
    for path in _iter_validated_paths(runs_dir):
        try:
            row = _normalize(path, runs_dir.parent, generation_indexes)
        except Exception as exc:
            skipped.append({"file": _manifest_path(path, manifest_root), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if row["id"] and row["statement"]:
            rows.append(row)
        else:
            skipped.append({"file": _manifest_path(path, manifest_root), "error": "missing id or statement"})

    deduped_rows, duplicate_rows = _dedupe(rows)
    required_columns = ["id", "statement", "answer", "solution", "verification_code"]
    complete_rows = [row for row in deduped_rows if all(row.get(column) for column in required_columns)]
    all_csv = output_dir / args.all_csv_name
    dedup_csv = output_dir / args.dedup_csv_name
    complete_csv = output_dir / args.complete_csv_name
    _write_csv(all_csv, rows)
    _write_csv(dedup_csv, deduped_rows)
    _write_csv(complete_csv, complete_rows)

    missing_counts_all = {
        column: sum(1 for row in rows if not row.get(column))
        for column in required_columns
    }
    missing_counts_dedup = {
        column: sum(1 for row in deduped_rows if not row.get(column))
        for column in required_columns
    }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_runs_dir": _manifest_path(runs_dir, manifest_root),
        "source_run_count": len({row["source_run"] for row in rows}),
        "all_row_count": len(rows),
        "dedup_row_count": len(deduped_rows),
        "complete_dedup_row_count": len(complete_rows),
        "duplicate_row_count": len(duplicate_rows),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "dedupe_key": ["id", "statement_sha256"],
        "complete_required_columns": required_columns,
        "missing_counts_all": missing_counts_all,
        "missing_counts_dedup": missing_counts_dedup,
        "columns": CSV_COLUMNS,
        "difficulty_label_counts_all": dict(Counter(row["difficulty_label"] or "unknown" for row in rows)),
        "operation_counts_all": dict(Counter(row["operation"] or "unknown" for row in rows)),
        "files": {
            "all_csv": _manifest_path(all_csv, manifest_root),
            "dedup_csv": _manifest_path(dedup_csv, manifest_root),
            "complete_csv": _manifest_path(complete_csv, manifest_root),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/public/validated_deep_runs_csv"))
    parser.add_argument("--manifest-root", type=Path, default=Path("."))
    parser.add_argument("--all-csv-name", default="entropymath_validated_deep_runs_all.csv")
    parser.add_argument("--dedup-csv-name", default="entropymath_validated_deep_runs_dedup.csv")
    parser.add_argument("--complete-csv-name", default="entropymath_validated_deep_runs_complete.csv")
    return parser


def main() -> None:
    manifest = export(_build_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
