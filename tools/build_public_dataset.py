#!/usr/bin/env python3
"""Package validated EntropyMath run artifacts into a release-candidate dataset.

The script reads a run directory produced by ``main.py`` and writes a compact
public dataset bundle with normalized JSON/JSONL records, a manifest, and a
short dataset README. It does not modify the source run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _problem_payloads(run_dir: Path) -> Iterable[Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
    validated_dir = run_dir / "validated_problems"
    if not validated_dir.exists():
        raise FileNotFoundError(f"validated_problems directory not found: {validated_dir}")
    for path in sorted(validated_dir.glob("*.json")):
        payload = _read_json(path)
        problem = payload.get("problem") if isinstance(payload, dict) else None
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if isinstance(problem, dict):
            yield path, problem, meta if isinstance(meta, dict) else {}


def _normalize_record(
    index: int,
    source_path: Path,
    problem: Dict[str, Any],
    meta: Dict[str, Any],
    *,
    dataset_name: str,
    include_answers: bool,
    include_solutions: bool,
    include_code: bool,
) -> Dict[str, Any]:
    statement = str(problem.get("statement") or problem.get("Question") or "").strip()
    answer = problem.get("answer", problem.get("Answer", ""))
    solution = str(problem.get("solution") or problem.get("Solution") or "").strip()
    code = str(problem.get("code") or problem.get("verification_code") or "").strip()
    problem_id = str(problem.get("id") or problem.get("ID") or f"{dataset_name}_{index:04d}")
    difficulty = problem.get("difficulty", problem.get("Difficulty", ""))
    difficulty_label = str(problem.get("difficulty_label") or "").strip().lower()
    if not difficulty_label:
        try:
            score = float(difficulty)
            if score < 5:
                difficulty_label = "easy"
            elif score < 7:
                difficulty_label = "medium"
            elif score < 9:
                difficulty_label = "hard"
            else:
                difficulty_label = "superhard"
        except (TypeError, ValueError):
            difficulty_label = ""
    op_type = str(problem.get("type") or problem.get("op_type") or "").strip()
    record: Dict[str, Any] = {
        "id": problem_id,
        "dataset": dataset_name,
        "index": index,
        "statement": statement,
        "difficulty": difficulty,
        "difficulty_label": difficulty_label,
        "type": op_type,
        "generation": meta.get("generation_count"),
        "parent_ids": problem.get("parent_ids", []),
        "ancestor_ids": problem.get("ancestor_ids", []),
        "statement_sha256": _sha256_text(statement),
        "answer_sha256": _sha256_text(str(answer)),
        "source": {
            "run_source": meta.get("source", ""),
            "run_generation_file": meta.get("generation_file", ""),
            "run_slot": meta.get("slot"),
            "validated_problem_file": source_path.name,
        },
    }
    if include_answers:
        record["answer"] = str(answer)
        record["answer_sha256"] = _sha256_text(str(answer))
    else:
        record["answer_sha256"] = _sha256_text(str(answer))
    if include_solutions:
        record["solution"] = solution
    if include_code:
        record["verification_code"] = code
    return record


def _dedupe(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seen_ids = set()
    seen_statements = set()
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for record in records:
        duplicate_reasons = []
        if record["id"] in seen_ids:
            duplicate_reasons.append("duplicate_id")
        if record["statement_sha256"] in seen_statements:
            duplicate_reasons.append("duplicate_statement")
        if duplicate_reasons:
            dropped.append(
                {
                    "id": record["id"],
                    "index": record["index"],
                    "reasons": duplicate_reasons,
                    "statement_sha256": record["statement_sha256"],
                }
            )
            continue
        seen_ids.add(record["id"])
        seen_statements.add(record["statement_sha256"])
        kept.append(record)
    return kept, dropped


def _counter(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    values = Counter(str(record.get(key, "") or "unknown") for record in records)
    return dict(sorted(values.items(), key=lambda item: item[0]))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = [
        _normalize_record(
            index,
            source_path,
            problem,
            meta,
            dataset_name=args.dataset_name,
            include_answers=not args.omit_answers,
            include_solutions=not args.omit_solutions,
            include_code=args.include_code,
        )
        for index, (source_path, problem, meta) in enumerate(_problem_payloads(run_dir), 1)
    ]
    records, dropped = _dedupe(raw_records)

    dataset_json = output_dir / "dataset.json"
    dataset_jsonl = output_dir / "dataset.jsonl"
    _write_json(dataset_json, records)
    dataset_jsonl.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    manifest = {
        "dataset_name": args.dataset_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "record_count": len(records),
        "raw_record_count": len(raw_records),
        "dropped_duplicate_count": len(dropped),
        "dropped_duplicates": dropped,
        "include_answers": not args.omit_answers,
        "include_solutions": not args.omit_solutions,
        "include_code": args.include_code,
        "license": args.license,
        "difficulty_counts": _counter(records, "difficulty_label"),
        "type_counts": _counter(records, "type"),
        "generation_counts": _counter(records, "generation"),
        "files": {
            "dataset_json": {
                "path": dataset_json.name,
                "sha256": _file_sha256(dataset_json),
            },
            "dataset_jsonl": {
                "path": dataset_jsonl.name,
                "sha256": _file_sha256(dataset_jsonl),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    record_fields = [
        "`id`",
        "`statement`",
        "`difficulty`",
        "`difficulty_label`",
        "`type`",
        "`generation`",
        "`parent_ids`",
        "`ancestor_ids`",
        "`statement_sha256`",
        "`answer_sha256`",
        "`source`",
    ]
    if not args.omit_answers:
        record_fields.insert(2, "`answer`")
    if not args.omit_solutions:
        record_fields.append("`solution`")
    if args.include_code:
        record_fields.append("`verification_code`")

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# {args.dataset_name}",
                "",
                "This is a release-candidate package generated from EntropyMath validated run artifacts.",
                "",
                "## Contents",
                "",
                "- `dataset.json`: normalized list of problem records",
                "- `dataset.jsonl`: one normalized problem record per line",
                "- `manifest.json`: counts, source run, duplicate drops, and file checksums",
                "",
                "## Record Fields",
                "",
                ", ".join(record_fields) + ".",
                "",
                f"Answers included: `{not args.omit_answers}`",
                f"Solutions included: `{not args.omit_solutions}`",
                f"Verification code included: `{args.include_code}`",
                f"License: `{args.license}`",
                "",
                "## Release Gate",
                "",
                "Before public release, each record should pass independent human mathematical review,",
                "answer verification, statement clarity review, and contamination/similarity screening.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing validated_problems/.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/public/entropymath_release_candidate"),
        help="Output directory for the public dataset bundle.",
    )
    parser.add_argument("--dataset-name", default="EntropyMath-Release-Candidate")
    parser.add_argument("--license", default="TBD")
    parser.add_argument("--omit-answers", action="store_true")
    parser.add_argument("--omit-solutions", action="store_true")
    parser.add_argument("--include-code", action="store_true")
    return parser


def main() -> None:
    manifest = build_dataset(_build_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
