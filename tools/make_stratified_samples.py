#!/usr/bin/env python3
"""Create deterministic stratified samples for model evaluation and human audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: List[Dict[str, str]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _release_id(row: Dict[str, str]) -> str:
    statement_hash = row.get("statement_sha256") or hashlib.sha256(row.get("statement", "").encode("utf-8")).hexdigest()
    return f"emv1_{statement_hash[:16]}"


def _with_release_ids(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        new_row = dict(row)
        new_row["release_id"] = _release_id(row)
        out.append(new_row)
    return out


def _columns(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    columns = list(rows[0].keys())
    if "release_id" in columns:
        columns = ["release_id"] + [column for column in columns if column != "release_id"]
    return columns


def _stratum(row: Dict[str, str]) -> str:
    return f"{row.get('difficulty_label') or 'unknown'}::{row.get('operation') or 'unknown'}"


def stratified_sample(rows: List[Dict[str, str]], n: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[_stratum(row)].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: (r.get("source_run", ""), r.get("id", ""), r.get("statement_sha256", "")))
        rng.shuffle(bucket)

    selected: List[Dict[str, str]] = []
    bucket_names = sorted(buckets)
    base = n // max(1, len(bucket_names))
    remainder = n % max(1, len(bucket_names))
    for idx, name in enumerate(bucket_names):
        take = min(len(buckets[name]), base + (1 if idx < remainder else 0))
        selected.extend(buckets[name][:take])

    if len(selected) < n:
        seen = {row["statement_sha256"] for row in selected}
        leftovers = [row for row in rows if row.get("statement_sha256") not in seen]
        leftovers.sort(key=lambda r: (r.get("source_run", ""), r.get("id", ""), r.get("statement_sha256", "")))
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])

    selected = selected[:n]
    selected.sort(key=lambda r: (_stratum(r), r.get("source_run", ""), r.get("id", "")))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("data/public/validated_deep_runs_csv/entropymath_validated_deep_runs_complete.csv"),
    )
    parser.add_argument(
        "--audit-source-csv",
        type=Path,
        default=None,
        help="Optional source CSV for the audit sample. Use the quality-gated release CSV when regenerating audit rows.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/public/evaluation_samples"))
    parser.add_argument("--model-n", type=int, default=120)
    parser.add_argument("--audit-n", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Preserve the existing model-evaluation sample and regenerate only the audit sample.",
    )
    args = parser.parse_args()

    rows = _with_release_ids(_read_rows(args.source_csv))
    audit_source = args.audit_source_csv or args.source_csv
    audit_base_rows = _with_release_ids(_read_rows(audit_source))

    if not args.skip_model:
        model_columns = _columns(rows)
        model_rows = stratified_sample(rows, args.model_n, args.seed)
        _write_rows(args.output_dir / "model_eval_stratified_120.csv", model_rows, model_columns)
        print(f"model_eval_rows={len(model_rows)}")
    else:
        print("model_eval_rows=preserved")

    audit_columns = _columns(audit_base_rows)
    audit_rows = stratified_sample(audit_base_rows, args.audit_n, args.seed + 1)
    _write_rows(args.output_dir / "human_audit_stratified_180.csv", audit_rows, audit_columns)
    print(f"human_audit_rows={len(audit_rows)}")


if __name__ == "__main__":
    main()
