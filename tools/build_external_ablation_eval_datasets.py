#!/usr/bin/env python3
"""Build external-ablation control and treatment JSONL arms for model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


REPO = Path(__file__).resolve().parents[1]
SEED_DIR = REPO / "data/seed/external_ablation"
DEFAULT_OUT = REPO / "data/eval/external_ablation"
BENCHES = ("math500", "aime2025", "gsm8k")
DEFAULT_EXPANDED_TARGET = 250
PUBLISHED_TREATMENT_DIR = REPO / "data/eval/external"
RECOVERED_PARTIAL_RUN_DIRS = (
    REPO / "data/runs/20260501-104151-deep-run",
    REPO / "data/runs/20260501-105457-deep-run",
    REPO / "data/runs/20260501-110419-deep-run",
)

HARD_TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("solution_provided_answer_incorrect", re.compile(r"\bprovided\s+(?:final\s+)?answer\b.{0,120}\bincorrect\b", re.I | re.S)),
    ("solution_candidate_answer_incorrect", re.compile(r"\bcandidate(?:\s+answer)?\b.{0,120}\bincorrect\b", re.I | re.S)),
    ("solution_final_answer_incorrect", re.compile(r"\bfinal\s+answer\b.{0,120}\bincorrect\b", re.I | re.S)),
)
SUPPORT_TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("support_sandbox_evidence", re.compile(r"\bsandbox\s+evidence\b", re.I)),
    ("support_sandbox_output", re.compile(r"\bsandbox\s+output\b", re.I)),
    ("support_tool_evidence", re.compile(r"\btool\s+evidence\b", re.I)),
    ("support_wait_recheck", re.compile(r"\bwait[,;:]?\s+", re.I)),
    ("support_re_evaluating", re.compile(r"\bre[- ]?evaluat(?:e|ing|ion)\b", re.I)),
    ("support_re_checking", re.compile(r"\bre[- ]?check(?:ing|ed)?\b", re.I)),
    ("support_appears", re.compile(r"\bappears\s+(?:to|that)\b", re.I)),
    ("support_suggests", re.compile(r"\bsuggests?\b", re.I)),
    ("support_implies_different", re.compile(r"\bimplies?\s+a\s+different\b", re.I)),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _statement_hash(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    return str(path.relative_to(REPO) if path.is_relative_to(REPO) else path)


def _norm_answer(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = text.strip().strip("$").strip().replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _numeric_equivalent(a: object, b: object, tol: float = 1e-6) -> bool:
    from fractions import Fraction

    a_text = str(a or "").strip()
    b_text = str(b or "").strip()
    if not a_text or not b_text:
        return False
    try:
        return abs(float(Fraction(a_text)) - float(Fraction(b_text))) <= tol
    except Exception:
        pass
    try:
        return abs(float(a_text) - float(b_text)) <= tol
    except Exception:
        return False


def _answers_equal(a: object, b: object) -> bool:
    return _norm_answer(a) == _norm_answer(b) or _numeric_equivalent(a, b)


def _quality_gate_text(question: str, solution: str) -> tuple[str, list[str]]:
    text = "\n".join([str(question or ""), str(solution or "")])
    hard = [name for name, pattern in HARD_TEXT_PATTERNS if pattern.search(text)]
    support = [name for name, pattern in SUPPORT_TEXT_PATTERNS if pattern.search(text)]
    if hard:
        return "excluded", sorted(set(hard + support))
    if support:
        return "excluded", sorted(set(support))
    return "kept", ["no_quality_gate_flag"]


def _quality_gate_problem(problem: Dict[str, Any], question: str, solution: str, answer: str) -> tuple[str, list[str], str]:
    status, reasons = _quality_gate_text(question, solution)
    runtime_status = "not_available"
    evidence = problem.get("execution_evidence") or {}
    canonical = evidence.get("canonical_answer") if isinstance(evidence, dict) else None
    if canonical not in (None, ""):
        runtime_status = "stored_execution_evidence"
        if not _answers_equal(canonical, answer):
            reasons = sorted(set(reasons + ["stored_execution_answer_disagreement"]))
            status = "excluded"
    elif problem.get("code"):
        runtime_status = "code_present_without_stored_execution_evidence"
    return status, reasons, runtime_status


def _seed_to_row(seed: Dict[str, Any], bench: str) -> Dict[str, Any]:
    return {
        "id": seed["ID"],
        "question": seed["Question"],
        "answer": str(seed["Answer"]),
        "solution": seed.get("Solution", ""),
        "difficulty": seed.get("Difficulty", ""),
        "_source_benchmark": bench,
        "_source_id": seed.get("_source_id", ""),
        "_source_dataset": seed.get("_source_dataset", ""),
        "_source_split": seed.get("_source_split", ""),
        "_ablation_seed_split": seed.get("_ablation_seed_split", ""),
        "_arm": "control",
        "_provenance_status": "control_seed",
    }


def _problem(payload: Dict[str, Any]) -> Dict[str, Any]:
    return dict(payload.get("problem", {}) or {})


def _validated_to_row(
    payload: Dict[str, Any],
    bench: str,
    run_dir: Path,
    provenance_status: str = "generated_full",
) -> Dict[str, Any]:
    problem = _problem(payload)
    meta = dict(payload.get("meta", {}) or {})
    parent_ids = list(problem.get("parent_ids", []) or [])
    question = problem.get("statement", "")
    answer = str(problem.get("answer", ""))
    solution = problem.get("solution", "")
    gate_status, gate_reasons, runtime_status = _quality_gate_problem(problem, question, solution, answer)
    return {
        "id": problem.get("id", ""),
        "question": question,
        "answer": answer,
        "solution": solution,
        "difficulty": problem.get("difficulty_label", ""),
        "_source_benchmark": bench,
        "_arm": "treatment",
        "_provenance_status": provenance_status,
        "_quality_gate_status": gate_status,
        "_quality_gate_reasons": ";".join(gate_reasons),
        "_quality_gate_runtime_status": runtime_status,
        "_op_type": problem.get("op_type", ""),
        "_parent_ids": parent_ids,
        "_generation": int(meta.get("generation_count", 0) or 0),
        "_target_diff": problem.get("target_diff"),
        "_run_dir": _display_path(run_dir),
        "_statement_sha256": _statement_hash(question),
    }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _latest_published_treatment_path(bench: str) -> Optional[Path]:
    files = sorted(PUBLISHED_TREATMENT_DIR.glob(f"{bench}_treatment_*.jsonl"))
    if not files:
        return None
    return max(files, key=lambda path: sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))


def read_published_treatment_rows(bench: str) -> List[Dict[str, Any]]:
    path = _latest_published_treatment_path(bench)
    if path is None:
        return []
    rows: List[Dict[str, Any]] = []
    for idx, raw in enumerate(_read_jsonl(path)):
        row = dict(raw)
        row.setdefault("_source_benchmark", bench)
        row["_arm"] = "treatment"
        row["_provenance_status"] = "published_existing"
        row["_source_file"] = _display_path(path)
        row["_source_row_index"] = idx
        row["_statement_sha256"] = row.get("_statement_sha256") or _statement_hash(row.get("question", ""))
        gate_status, gate_reasons = _quality_gate_text(row.get("question", ""), row.get("solution", ""))
        row["_quality_gate_status"] = gate_status
        row["_quality_gate_reasons"] = ";".join(gate_reasons)
        row["_quality_gate_runtime_status"] = "not_available"
        rows.append(row)
    return rows


def build_controls(bench: str, out_dir: Path) -> Dict[str, Any]:
    seeds = _read_json(SEED_DIR / f"{bench}_ablation20_seeds.json")
    rows = [_seed_to_row(seed, bench) for seed in seeds]
    existing = [row for row in rows if row.get("_ablation_seed_split") == "existing_matched"]
    new = [row for row in rows if row.get("_ablation_seed_split") == "new_diagnostic"]
    if len(existing) != 10 or len(new) != 10:
        raise ValueError(f"{bench}: expected 10 existing and 10 new seeds; got {len(existing)} and {len(new)}")

    paths = {
        "existing_control": out_dir / f"{bench}_existing_control_10.jsonl",
        "new_control": out_dir / f"{bench}_new_control_10.jsonl",
        "combined_control": out_dir / f"{bench}_control_20.jsonl",
    }
    counts = {
        "existing_control": _jsonl_write(paths["existing_control"], existing),
        "new_control": _jsonl_write(paths["new_control"], new),
        "combined_control": _jsonl_write(paths["combined_control"], rows),
    }
    return {"paths": {key: str(path.relative_to(REPO)) for key, path in paths.items()}, "counts": counts}


def build_treatment(bench: str, run_dirs: List[Path], out_dir: Path, label: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for run_dir in run_dirs:
        resolved = run_dir if run_dir.is_absolute() else REPO / run_dir
        vp_dir = resolved / "validated_problems"
        if not vp_dir.is_dir():
            continue
        for path in sorted(vp_dir.glob("*.json")):
            try:
                payload = _read_json(path)
            except Exception:
                continue
            row = _validated_to_row(payload, bench, resolved)
            key = row.get("_statement_sha256") or _statement_hash(row.get("question", ""))
            if not row.get("question") or key in seen or row.get("_quality_gate_status") != "kept":
                continue
            seen.add(key)
            rows.append(row)

    out = out_dir / f"{bench}_{label}_treatment_{len(rows)}.jsonl"
    _jsonl_write(out, rows)
    return {"path": str(out.relative_to(REPO)), "count": len(rows)}


def _manifest_seed_sets() -> tuple[Dict[str, set[str]], Dict[str, set[str]]]:
    manifest = yaml.safe_load((SEED_DIR / "combo_manifest_ablation.yaml").read_text(encoding="utf-8"))
    existing: Dict[str, set[str]] = {}
    new: Dict[str, set[str]] = {}
    for bench in BENCHES:
        existing[bench] = set(manifest[bench]["existing_all"]["seed_ids"])
        new[bench] = set(manifest[bench]["new_all"]["seed_ids"])
    return existing, new


def discover_new_full_run_dirs(bench: str) -> List[Path]:
    existing_ids, new_ids = _manifest_seed_sets()
    out: List[Path] = []
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
        seed_ids = set(result.get("seed_ids") or [])
        if not seed_ids or not seed_ids <= new_ids[bench]:
            continue
        condition = str((result.get("parameters") or {}).get("ablation_condition") or options.get("ablation_condition") or "full")
        if condition != "full":
            continue
        out.append(run_dir)
    return out


def _infer_recovered_benchmark(run_dir: Path) -> str:
    vp_dir = run_dir / "validated_problems"
    for path in sorted(vp_dir.glob("*.json")) if vp_dir.is_dir() else []:
        try:
            problem = _problem(_read_json(path))
        except Exception:
            continue
        text = " ".join([str(problem.get("id", "")), str(problem.get("parent_ids", "")), str(problem.get("statement", ""))])
        for bench in BENCHES:
            if bench in text:
                return bench
    return ""


def _iter_recovered_partial_run_dirs(bench: str) -> List[Path]:
    candidates = set(RECOVERED_PARTIAL_RUN_DIRS)
    runs_root = REPO / "data/runs"
    for run_dir in sorted(runs_root.iterdir()) if runs_root.is_dir() else []:
        if (run_dir / "full_run_result.json").exists():
            continue
        vp_dir = run_dir / "validated_problems"
        if not vp_dir.is_dir():
            continue
        has_full_ablation_problem = False
        for path in sorted(vp_dir.glob("*.json")):
            try:
                problem = _problem(_read_json(path))
            except Exception:
                continue
            if problem.get("ablation_condition") == "full":
                has_full_ablation_problem = True
                break
        if has_full_ablation_problem:
            candidates.add(run_dir)
    return [run_dir for run_dir in sorted(candidates) if run_dir.exists() and _infer_recovered_benchmark(run_dir) == bench]


def recovered_partial_rows(bench: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for run_dir in _iter_recovered_partial_run_dirs(bench):
        vp_dir = run_dir / "validated_problems"
        for path in sorted(vp_dir.glob("*.json")):
            try:
                payload = _read_json(path)
            except Exception as exc:
                manifest_rows.append(
                    {
                        "benchmark": bench,
                        "run_dir": _display_path(run_dir),
                        "validated_problem": _display_path(path),
                        "quality_gate_status": "excluded",
                        "quality_gate_reasons": f"json_error:{exc}",
                    }
                )
                continue
            row = _validated_to_row(payload, bench, run_dir, provenance_status="partial_recovered")
            row["_source_file"] = _display_path(path)
            rows.append(row)
            manifest_rows.append(
                {
                    "benchmark": bench,
                    "run_dir": _display_path(run_dir),
                    "validated_problem": _display_path(path),
                    "statement_sha256": row.get("_statement_sha256", ""),
                    "quality_gate_status": row.get("_quality_gate_status", ""),
                    "quality_gate_reasons": row.get("_quality_gate_reasons", ""),
                    "quality_gate_runtime_status": row.get("_quality_gate_runtime_status", ""),
                }
            )
    return rows, manifest_rows


def collect_expanded_full_rows(bench: str) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    excluded_count = 0
    source_counts: Dict[str, int] = {}
    recovered_manifest: List[Dict[str, Any]] = []

    sources: List[Tuple[str, List[Dict[str, Any]]]] = [("published_existing", read_published_treatment_rows(bench))]
    generated_rows: List[Dict[str, Any]] = []
    for run_dir in discover_new_full_run_dirs(bench):
        for path in sorted((run_dir / "validated_problems").glob("*.json")):
            try:
                generated_rows.append(_validated_to_row(_read_json(path), bench, run_dir, "generated_full"))
            except Exception:
                continue
    sources.append(("generated_full", generated_rows))
    recovered_rows, recovered_manifest = recovered_partial_rows(bench)
    sources.append(("partial_recovered", recovered_rows))

    for source, source_rows in sources:
        source_counts[source] = len(source_rows)
        for row in source_rows:
            key = row.get("_statement_sha256") or _statement_hash(row.get("question", ""))
            if not row.get("question") or row.get("_quality_gate_status") != "kept":
                excluded_count += 1
                continue
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            rows.append(row)

    summary = {
        "benchmark": bench,
        "source_counts_before_gate": source_counts,
        "quality_excluded_count": excluded_count,
        "dedup_excluded_count": duplicate_count,
        "pool_count": len(rows),
        "provenance_counts": {status: sum(1 for row in rows if row.get("_provenance_status") == status) for status in sorted({row.get("_provenance_status", "") for row in rows})},
    }
    return rows, summary, recovered_manifest


def _round_robin(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: (item.get("_run_dir", ""), item.get("_generation", 0), item.get("id", ""))):
        buckets.setdefault(str(row.get("_run_dir") or row.get("_source_file") or "unknown"), []).append(row)
    ordered: List[Dict[str, Any]] = []
    keys = sorted(buckets)
    while keys:
        next_keys: List[str] = []
        for key in keys:
            bucket = buckets[key]
            if bucket:
                ordered.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return ordered


def deterministic_cap(rows: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    if len(rows) <= target:
        return list(rows)
    published = [row for row in rows if row.get("_provenance_status") == "published_existing"]
    recovered = [row for row in rows if row.get("_provenance_status") == "partial_recovered"]
    generated = [row for row in rows if row.get("_provenance_status") not in {"published_existing", "partial_recovered"}]
    selected = sorted(published, key=lambda row: (row.get("_source_file", ""), int(row.get("_source_row_index", 0) or 0)))[:target]
    remaining = target - len(selected)
    if remaining <= 0:
        return selected
    selected.extend(sorted(recovered, key=lambda row: (row.get("_run_dir", ""), row.get("id", "")))[:remaining])
    remaining = target - len(selected)
    if remaining <= 0:
        return selected
    selected.extend(_round_robin(generated)[:remaining])
    return selected


def build_expanded_full_treatment(bench: str, out_dir: Path, target: int, fail_under_target: bool) -> Dict[str, Any]:
    rows, summary, recovered_manifest = collect_expanded_full_rows(bench)
    pool_path = out_dir / f"{bench}_expanded_full_pool_{len(rows)}.jsonl"
    _jsonl_write(pool_path, rows)
    exact_path = out_dir / f"{bench}_expanded_full_treatment_{target}.jsonl"
    exact_count = 0
    if len(rows) >= target:
        capped = deterministic_cap(rows, target)
        exact_count = _jsonl_write(exact_path, capped)
    elif exact_path.exists():
        exact_path.unlink()
    if recovered_manifest:
        manifest_path = out_dir / "recovered_partial_manifest.jsonl"
        existing = _read_jsonl(manifest_path) if manifest_path.exists() else []
        existing = [row for row in existing if row.get("benchmark") != bench]
        _jsonl_write(manifest_path, existing + recovered_manifest)
        summary["recovered_manifest"] = _display_path(manifest_path)
    if fail_under_target and len(rows) < target:
        summary.update(
            {
                "pool_path": _display_path(pool_path),
                "exact_path": _display_path(exact_path),
                "exact_count": exact_count,
                "target": target,
                "ready": False,
            }
        )
        raise ValueError(f"{bench}: only {len(rows)} quality-gated expanded treatment rows; need {target}")
    summary.update(
        {
            "pool_path": _display_path(pool_path),
            "exact_path": _display_path(exact_path),
            "exact_count": exact_count,
            "target": target,
            "ready": exact_count == target,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=BENCHES, action="append", help="Benchmark to build; defaults to all.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--treatment-run-dir",
        action="append",
        default=[],
        help="Saved run dir to include as treatment source. Can be repeated.",
    )
    parser.add_argument(
        "--auto-new-full-treatment",
        action="store_true",
        help="Auto-discover new_diagnostic/full external-ablation runs and write current treatment JSONLs.",
    )
    parser.add_argument("--treatment-label", default="new", help="Filename label for treatment rows from run dirs.")
    parser.add_argument("--build-expanded-full", action="store_true", help="Build published+new+recovered expanded treatment pool and exact target JSONL.")
    parser.add_argument("--expanded-target", type=int, default=DEFAULT_EXPANDED_TARGET)
    parser.add_argument("--allow-underfilled-expanded", action="store_true", help="Write pools and summaries even when exact target files cannot be built.")
    args = parser.parse_args()

    benches = args.bench or list(BENCHES)
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    summary: Dict[str, Any] = {"out_dir": str(out_dir.relative_to(REPO)), "benchmarks": {}}
    for bench in benches:
        entry = {"controls": build_controls(bench, out_dir)}
        treatment_dirs = [Path(raw) for raw in args.treatment_run_dir]
        treatment_label = args.treatment_label
        if args.auto_new_full_treatment:
            treatment_dirs = discover_new_full_run_dirs(bench)
            treatment_label = "new_full"
        if treatment_dirs:
            entry["treatment"] = build_treatment(
                bench,
                treatment_dirs,
                out_dir,
                treatment_label,
            )
        if args.build_expanded_full:
            entry["expanded_full"] = build_expanded_full_treatment(
                bench,
                out_dir,
                args.expanded_target,
                fail_under_target=not args.allow_underfilled_expanded,
            )
        summary["benchmarks"][bench] = entry
    if args.build_expanded_full:
        summary_path = out_dir / "expanded_full_summary.json"
        summary["expanded_full_summary"] = _display_path(summary_path)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
