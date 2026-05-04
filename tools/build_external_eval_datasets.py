#!/usr/bin/env python3
"""Convert EntropyMath generation results into eval-harness JSONL datasets.

Reads:
- data/seed/external/<bench>_seeds.json  -> control set (10 seeds)
- data/runs/<stamp>-deep-run/validated_problems/*.json  -> treatment source rows when saved runs are available

Writes:
- data/eval/external/<bench>_control_10.jsonl
- data/eval/external/<bench>_treatment.jsonl

Treatment metadata preserved for stratified analysis:
  op_type, difficulty_label, parent_ids, generation_count, _source_benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
SEED_DIR = REPO / "data/seed/external"
EVAL_DIR = REPO / "data/eval/external"


def _seed_to_jsonl_row(seed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": seed["ID"],
        "question": seed["Question"],
        "answer": str(seed["Answer"]),
        "solution": seed.get("Solution", ""),
        "difficulty": seed.get("Difficulty", ""),
        "_source_benchmark": seed.get("_source_benchmark", ""),
        "_source_id": seed.get("_source_id", ""),
        "_arm": "control",
    }


def _validated_to_jsonl_row(payload: Dict[str, Any], source_benchmark: str) -> Dict[str, Any]:
    p = payload.get("problem", {}) or {}
    meta = payload.get("meta", {}) or {}
    return {
        "id": p.get("id", ""),
        "question": p.get("statement", ""),
        "answer": str(p.get("answer", "")),
        "solution": p.get("solution", ""),
        "difficulty": p.get("difficulty_label", ""),
        "_source_benchmark": source_benchmark,
        "_arm": "treatment",
        # Stratification metadata
        "_op_type": p.get("op_type", ""),
        "_parent_ids": list(p.get("parent_ids", []) or []),
        "_generation": int(meta.get("generation_count", 0) or 0),
        "_target_diff": p.get("target_diff"),
    }


def build_control(bench: str) -> Path:
    seed_path = SEED_DIR / f"{bench}_seeds.json"
    if bench == "olympiadbench_en":
        seed_path = SEED_DIR / "olympiadbench_seeds.json"
    seeds = json.loads(seed_path.read_text())
    rows = [_seed_to_jsonl_row(s) for s in seeds]
    out = EVAL_DIR / f"{bench}_control_{len(rows)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def build_treatment(bench: str, run_dirs: List[Path]) -> Path:
    rows: List[Dict[str, Any]] = []
    seen_statements: Dict[str, str] = {}  # statement -> first id (for dedup)
    for run_dir in run_dirs:
        vp_dir = run_dir / "validated_problems"
        if not vp_dir.is_dir():
            print(f"  WARN: no validated_problems/ in {run_dir}")
            continue
        for jf in sorted(vp_dir.glob("*.json")):
            try:
                payload = json.loads(jf.read_text())
            except Exception as e:
                print(f"  skip {jf}: {e}")
                continue
            row = _validated_to_jsonl_row(payload, bench)
            stmt_key = (row.get("question", "") or "").strip()[:300]
            if not stmt_key or stmt_key in seen_statements:
                continue  # dedup by first 300 chars
            seen_statements[stmt_key] = row["id"]
            rows.append(row)
    out = EVAL_DIR / f"{bench}_treatment_{len(rows)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["math500", "aime2025", "olympiadbench_en", "gsm8k"])
    ap.add_argument("--run-dirs", nargs="+", default=[],
                    help="One or more saved run directories, absolute or relative to the repository root.")
    ap.add_argument("--control-only", action="store_true")
    args = ap.parse_args()

    print(f"=== {args.bench} ===")
    ctl = build_control(args.bench)
    print(f"  control → {ctl} ({sum(1 for _ in ctl.open())} rows)")
    if args.control_only:
        return
    if not args.run_dirs:
        print("  no run dirs supplied; treatment skipped")
        return
    run_dirs = [Path(p) if Path(p).is_absolute() else REPO / p for p in args.run_dirs]
    trt = build_treatment(args.bench, run_dirs)
    n = sum(1 for _ in trt.open())
    print(f"  treatment → {trt} ({n} rows)")


if __name__ == "__main__":
    main()
