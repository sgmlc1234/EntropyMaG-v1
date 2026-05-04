#!/usr/bin/env python3
"""Repair model labels in flattened model-evaluation summaries.

The evaluation harness emitted "." as the model key in per-model summaries.
For reviewer traceability, the canonical model identifier is recovered from
each run_manifest.json and written back into the per-model summary JSON,
per-model summary_flat.csv, and combined_summary_flat.csv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "analysis/model_eval_results/direct_no_tool"
SUMMARY_FIELDS = [
    "model",
    "group_type",
    "group",
    "problem_count",
    "run_count",
    "accuracy_at_1",
    "accuracy_at_1_ci95",
    "pass_at_k",
    "pass_at_k_ci95",
    "consistency_gap",
    "answer_extraction_failure_rate",
    "gold_missing_rate",
    "tool_use_rate",
    "python_error_rate",
]


def canonical_model_id(model_dir: Path) -> str:
    manifest = json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8"))
    model_id = (((manifest.get("model") or {}).get("model")) or "").strip()
    if not model_id:
        raise ValueError(f"Missing model.model in {model_dir / 'run_manifest.json'}")
    return model_id


def relabel_summary_json(model_dir: Path, model_id: str) -> None:
    path = model_dir / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models") or {}
    if list(models.keys()) == [model_id]:
        return
    if "." not in models:
        raise ValueError(f"Expected '.' model key in {path}, found {list(models)}")
    payload["models"] = {model_id: models["."]}
    payload["model_count"] = 1
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relabel_summary_flat(model_dir: Path, model_id: str) -> list[dict[str, str]]:
    path = model_dir / "summary_flat.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["model"] = model_id
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def model_dirs() -> list[Path]:
    return sorted(path.parent for path in EVAL_DIR.glob("*/*/run_manifest.json"))


def main() -> None:
    combined: list[dict[str, str]] = []
    ids: list[str] = []
    for model_dir in model_dirs():
        model_id = canonical_model_id(model_dir)
        ids.append(model_id)
        relabel_summary_json(model_dir, model_id)
        combined.extend(relabel_summary_flat(model_dir, model_id))

    combined_path = EVAL_DIR / "combined_summary_flat.csv"
    with combined_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(combined)

    print(json.dumps({"models": ids, "rows": len(combined), "combined": str(combined_path)}, indent=2))


if __name__ == "__main__":
    main()
