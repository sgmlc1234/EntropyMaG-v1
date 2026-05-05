#!/usr/bin/env python3
"""Summarize EntropyMath ablation micro-study runs.

The analyzer is intentionally local and text/code based by default. It can run
an optional API-backed shadow solvability pass, but the default path only uses
saved run artifacts and sandbox execution evidence.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "data/seed/external_ablation/combo_manifest_ablation.yaml"
DEFAULT_OUT = REPO / "data/analysis/ablation_microstudy"
DEFAULT_CONDITIONS = ("no_near_copy", "no_solvability")
_VALIDATION_WORKER_MODULE = None

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_STATEMENT_NORMALIZE_RE = re.compile(r"\s+")
_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-zA-Z]+")
_COMMON_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "is",
        "are",
        "be",
        "with",
        "that",
        "this",
        "these",
        "those",
        "as",
        "at",
        "by",
        "let",
        "find",
        "compute",
        "determine",
        "prove",
        "show",
        "given",
        "suppose",
        "all",
        "any",
        "some",
        "each",
        "every",
        "if",
        "then",
        "we",
        "call",
        "say",
        "define",
        "consider",
        "problem",
        "solution",
        "answer",
        "such",
        "where",
        "which",
        "from",
    }
)


def _normalize_answer(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_statement(text: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z\s]", " ", (text or "").lower())
    return _STATEMENT_NORMALIZE_RE.sub(" ", cleaned).strip()


def _statement_sha256(text: str) -> str:
    return hashlib.sha256(_normalize_statement(text).encode("utf-8")).hexdigest()


def _salient_tokens(text: str, min_len: int = 3, limit: int = 64) -> List[str]:
    tokens = [tok for tok in _TOKEN_SPLIT_RE.split((text or "").lower()) if tok]

    def gather(min_length: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for tok in tokens:
            if tok in _COMMON_STOPWORDS or tok.isdigit() or len(tok) < min_length:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= limit:
                break
        return out

    strict = gather(min_len)
    return strict or gather(2)


def _jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = set(a), set(b)
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _last_numeric(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", lines[-1].replace(",", ""))
    return matches[-1] if matches else ""


def _code_assessment(problem: Dict[str, Any], *, execute_missing: bool = True) -> Dict[str, Any]:
    cached = problem.get("_code_execution_assessment")
    if isinstance(cached, dict):
        return dict(cached)
    code = str(problem.get("code") or "").strip()
    if not code:
        return {
            "ok": False,
            "execution_ok": False,
            "answer_match": False,
            "canonical_answer": "",
            "error_type": "missing_code",
            "validation_reason": "No code provided",
        }
    if not execute_missing:
        return {
            "ok": True,
            "execution_ok": True,
            "answer_match": None,
            "canonical_answer": "",
            "error_type": "",
            "validation_reason": "Code execution skipped by analyzer option.",
        }
    try:
        from deepagent.python_sandbox import execute_python_code
    except ModuleNotFoundError as exc:
        return {
            "ok": False,
            "execution_ok": False,
            "answer_match": False,
            "canonical_answer": "",
            "error_type": "sandbox_import_error",
            "validation_reason": f"Could not import DeepAgent sandbox: {exc}",
        }
    outcome = execute_python_code(code, mode=problem.get("code_runtime_mode"), trace_enabled=False)
    if not outcome.get("ok"):
        return {
            "ok": False,
            "execution_ok": False,
            "answer_match": False,
            "canonical_answer": "",
            "error_type": outcome.get("error_type", "runtime_error"),
            "validation_reason": outcome.get("error_message", ""),
            "outcome": outcome,
        }
    stdout = (outcome.get("stdout") or "").strip()
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    canonical = _normalize_answer(lines[-1]) if lines else _normalize_answer(stdout)
    if not canonical:
        canonical = _normalize_answer(_last_numeric(stdout))
    expected = _normalize_answer(problem.get("answer", ""))
    answer_match = bool(expected and canonical == expected)
    return {
        "ok": bool(canonical),
        "execution_ok": bool(canonical),
        "answer_match": answer_match,
        "canonical_answer": canonical,
        "error_type": "" if canonical else "answer_mismatch",
        "validation_reason": "Analyzer sandbox execution.",
        "outcome": outcome,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _problem_id(problem: Dict[str, Any]) -> str:
    return str(problem.get("id") or problem.get("ID") or "")


def _problem_statement(problem: Dict[str, Any]) -> str:
    return str(problem.get("statement") or problem.get("Question") or "")


def _problem_answer(problem: Dict[str, Any]) -> str:
    return str(problem.get("answer") if "answer" in problem else problem.get("Answer", ""))


def _load_seed_lookup(seed_spec: str) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    if not seed_spec:
        return lookup
    for raw in [part.strip() for part in seed_spec.split(",") if part.strip()]:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            continue
        for row in _load_json(path):
            normalized = dict(row)
            normalized.setdefault("id", row.get("ID"))
            normalized.setdefault("statement", row.get("Question"))
            normalized.setdefault("answer", str(row.get("Answer", "")))
            lookup[_problem_id(normalized)] = normalized
    return lookup


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _manifest_lookup(manifest: Dict[str, Any]) -> Dict[frozenset, Dict[str, str]]:
    out: Dict[frozenset, Dict[str, str]] = {}
    for bench, bench_entry in manifest.items():
        if not isinstance(bench_entry, dict):
            continue
        for combo, entry in bench_entry.items():
            if combo == "seed_file" or not isinstance(entry, dict):
                continue
            seed_ids = entry.get("seed_ids") or []
            out[frozenset(seed_ids)] = {
                "benchmark": bench,
                "combo": combo,
                "seed_split": entry.get("split", ""),
                "target_treatment_role": entry.get("target_treatment_role", ""),
            }
    return out


def _infer_run_labels(summary: Dict[str, Any], combo_lookup: Dict[frozenset, Dict[str, str]]) -> Dict[str, str]:
    seed_ids = [str(seed_id) for seed_id in summary.get("seed_ids", []) or []]
    labels = dict(combo_lookup.get(frozenset(seed_ids), {}))
    if not labels.get("benchmark"):
        prefix = seed_ids[0].split("_", 1)[0] if seed_ids else "unknown"
        labels.setdefault("benchmark", prefix)
        labels.setdefault("combo", "unknown")
        labels.setdefault("seed_split", "unknown")
        labels.setdefault("target_treatment_role", "unknown")
    params = summary.get("parameters", {}) or {}
    options = summary.get("session_options", {}) or {}
    labels["condition"] = str(params.get("ablation_condition") or options.get("ablation_condition") or "full")
    labels["max_generations"] = str(options.get("max_generations", ""))
    labels["target_problem_count"] = str(options.get("target_problem_count") or "")
    return labels


def _load_parent_lookup(run_dir: Path, summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup = _load_seed_lookup((summary.get("session_options", {}) or {}).get("seed_spec", ""))
    for generation_file in sorted(run_dir.glob("generation_*.json")):
        try:
            rows = _load_json(generation_file)
        except Exception:
            continue
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and _problem_id(row):
                    lookup[_problem_id(row)] = row
    for payload_path in sorted((run_dir / "validated_problems").glob("*.json")):
        try:
            payload = _load_json(payload_path)
        except Exception:
            continue
        problem = payload.get("problem", {}) if isinstance(payload, dict) else {}
        if _problem_id(problem):
            lookup[_problem_id(problem)] = problem
    final_state = summary.get("final_state", {}) or {}
    for key in ["current_generation", "candidates", "failed_problems"]:
        for problem in final_state.get(key, []) or []:
            if isinstance(problem, dict) and _problem_id(problem):
                lookup[_problem_id(problem)] = problem
    for item in final_state.get("approved_candidates", []) or []:
        problem = (item or {}).get("problem", {})
        if isinstance(problem, dict) and _problem_id(problem):
            lookup[_problem_id(problem)] = problem
    return lookup


def _candidate_records(run_dir: Path, summary: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    records: List[Tuple[str, Dict[str, Any]]] = []
    for payload_path in sorted((run_dir / "validated_problems").glob("*.json")):
        try:
            payload = _load_json(payload_path)
        except Exception:
            continue
        problem = payload.get("problem", {}) if isinstance(payload, dict) else {}
        if problem:
            records.append(("retained", problem))
    for problem in (summary.get("final_state", {}) or {}).get("failed_problems", []) or []:
        if isinstance(problem, dict) and _problem_statement(problem):
            records.append(("terminal_failed", problem))
    return records


def _parent_metrics(problem: Dict[str, Any], parent_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    child_statement = _problem_statement(problem)
    child_norm = _normalize_statement(child_statement)
    child_answer = _normalize_answer(_problem_answer(problem))
    child_tokens = _salient_tokens(child_statement)
    best_similarity = 0.0
    best_parent_id = ""
    best_jaccard = 0.0
    same_answer = False
    for parent_id in problem.get("parent_ids", []) or []:
        parent = parent_lookup.get(parent_id)
        if not parent:
            continue
        parent_norm = _normalize_statement(_problem_statement(parent))
        if parent_norm:
            sim = difflib.SequenceMatcher(None, parent_norm, child_norm).ratio()
            if sim > best_similarity:
                best_similarity = sim
                best_parent_id = parent_id
        best_jaccard = max(best_jaccard, _jaccard_similarity(child_tokens, _salient_tokens(_problem_statement(parent))))
        if child_answer and _normalize_answer(_problem_answer(parent)) == child_answer:
            same_answer = True
    near_copy = bool(same_answer and best_similarity >= 0.85)
    return {
        "parent_max_sequence_similarity": round(best_similarity, 6),
        "parent_min_jaccard_distance": round(1.0 - best_jaccard, 6),
        "matched_parent_id": best_parent_id,
        "parent_answer_same": same_answer,
        "near_copy_candidate": near_copy,
    }


def _maybe_shadow_solvability(problem: Dict[str, Any], run_shadow: bool) -> Dict[str, Any]:
    assessment = dict(problem.get("solvability_assessment") or {})
    if run_shadow and assessment.get("ablation_skipped"):
        assess_solvability = _load_assess_solvability()
        assessment = assess_solvability(problem, invariant_bundles=None)
    return {
        "solvability_required": bool(assessment.get("gate_required", False)),
        "shadow_solvability_checked": bool(assessment) and not bool(assessment.get("ablation_skipped", False)),
        "shadow_solvability_fail": assessment.get("verdict") == "fail",
        "solvability_verdict": assessment.get("verdict", ""),
        "solvability_failure_type": assessment.get("failure_type", ""),
    }


def _load_assess_solvability():
    """Load the validation worker without importing validation package wiring."""
    global _VALIDATION_WORKER_MODULE
    if _VALIDATION_WORKER_MODULE is None:
        worker_path = REPO / "deepagent/nodes/validation/worker.py"
        spec = importlib.util.spec_from_file_location(
            "_entropymath_ablation_validation_worker",
            worker_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load validation worker from {worker_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _VALIDATION_WORKER_MODULE = module
    return _VALIDATION_WORKER_MODULE.assess_solvability


def _percent(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return round(100.0 * num / den, 3)


def _aggregate(rows: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)

    out: List[Dict[str, Any]] = []
    for key_values, group in sorted(grouped.items()):
        n = len(group)
        retained = sum(1 for row in group if row["candidate_source"] == "retained")
        near_copy = sum(1 for row in group if row["near_copy_candidate"])
        same_answer = sum(1 for row in group if row["parent_answer_same"])
        code_errors = sum(1 for row in group if row["code_execution_error"])
        mismatches = sum(1 for row in group if row["answer_mismatch"])
        required = sum(1 for row in group if row["solvability_required"])
        shadow_checked = sum(1 for row in group if row["shadow_solvability_checked"])
        shadow_fail = sum(1 for row in group if row["shadow_solvability_fail"])
        critical = sum(1 for row in group if row["critical_error_proxy"])
        statement_unique = len({_statement_sha256(row["statement"]) for row in group if row["statement"]})
        result = {name: value for name, value in zip(keys, key_values)}
        result.update(
            {
                "candidate_count": n,
                "retained_count": retained,
                "statement_unique_count": statement_unique,
                "validator_pass_yield_pct": _percent(retained, n),
                "near_copy_candidate_count": near_copy,
                "near_copy_candidate_rate_pct": _percent(near_copy, n),
                "mean_parent_similarity": round(
                    sum(float(row["parent_max_sequence_similarity"]) for row in group) / max(1, n),
                    6,
                ),
                "mean_parent_min_jaccard_distance": round(
                    sum(float(row["parent_min_jaccard_distance"]) for row in group) / max(1, n),
                    6,
                ),
                "statement_unique_rate_pct": _percent(statement_unique, n),
                "parent_answer_same_count": same_answer,
                "parent_answer_same_rate_pct": _percent(same_answer, n),
                "code_execution_error_count": code_errors,
                "code_execution_error_rate_pct": _percent(code_errors, n),
                "answer_mismatch_count": mismatches,
                "answer_mismatch_rate_pct": _percent(mismatches, n),
                "solvability_required_count": required,
                "solvability_required_rate_pct": _percent(required, n),
                "shadow_solvability_checked_count": shadow_checked,
                "shadow_solvability_fail_count": shadow_fail,
                "shadow_solvability_fail_rate_pct": _percent(shadow_fail, shadow_checked),
                "critical_error_proxy_count": critical,
                "critical_error_proxy_rate_pct": _percent(critical, n),
            }
        )
        out.append(result)
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_run(run_dir: Path, combo_lookup: Dict[frozenset, Dict[str, str]], run_shadow: bool, execute_code: bool) -> List[Dict[str, Any]]:
    summary = _load_json(run_dir / "full_run_result.json")
    labels = _infer_run_labels(summary, combo_lookup)
    parent_lookup = _load_parent_lookup(run_dir, summary)
    try:
        run_dir_label = str(run_dir.relative_to(REPO))
    except ValueError:
        run_dir_label = str(run_dir)
    rows: List[Dict[str, Any]] = []
    for source, problem in _candidate_records(run_dir, summary):
        parent = _parent_metrics(problem, parent_lookup)
        code = _code_assessment(problem, execute_missing=execute_code)
        solvability = _maybe_shadow_solvability(problem, run_shadow)
        code_error = not bool(code.get("execution_ok", code.get("ok", False)))
        answer_mismatch = bool(code.get("execution_ok", False)) and code.get("answer_match") is False
        critical = code_error or answer_mismatch or bool(solvability["shadow_solvability_fail"])
        rows.append(
            {
                **labels,
                "run_dir": run_dir_label,
                "candidate_source": source,
                "id": _problem_id(problem),
                "statement": _problem_statement(problem),
                "statement_sha256": _statement_sha256(_problem_statement(problem)),
                "answer": _problem_answer(problem),
                "op_type": problem.get("op_type", ""),
                "generation": (problem.get("generation_meta", {}) or {}).get("generation_count", ""),
                "parent_ids": ";".join(problem.get("parent_ids", []) or []),
                **parent,
                "code_execution_error": code_error,
                "answer_mismatch": answer_mismatch,
                "code_error_type": code.get("error_type", ""),
                "canonical_answer": code.get("canonical_answer", ""),
                **solvability,
                "critical_error_proxy": critical,
            }
        )
    return rows


def _discover_external_ablation_run_dirs(conditions: Iterable[str]) -> List[Path]:
    wanted = {str(condition) for condition in conditions}
    run_dirs: List[Path] = []
    runs_root = REPO / "data/runs"
    for run_dir in sorted(runs_root.iterdir()) if runs_root.exists() else []:
        result_path = run_dir / "full_run_result.json"
        if not result_path.exists():
            continue
        try:
            summary = _load_json(result_path)
        except Exception:
            continue
        options = summary.get("session_options", {}) or {}
        if "external_ablation" not in str(options.get("seed_spec") or ""):
            continue
        params = summary.get("parameters", {}) or {}
        condition = str(params.get("ablation_condition") or options.get("ablation_condition") or "full")
        if wanted and condition not in wanted:
            continue
        run_dirs.append(run_dir)
    return run_dirs


def _condition_counts(rows: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("benchmark", "")), str(row.get("condition", "")))].append(row)
    out: List[Dict[str, Any]] = []
    for benchmark, condition in sorted(grouped):
        group = grouped[(benchmark, condition)]
        statement_unique = len({row["statement_sha256"] for row in group if row.get("statement_sha256")})
        out.append(
            {
                "benchmark": benchmark,
                "condition": condition,
                "target_statement_unique_candidates": target,
                "candidate_count": len(group),
                "retained_count": sum(1 for row in group if row["candidate_source"] == "retained"),
                "statement_unique_count": statement_unique,
                "remaining_to_target": max(0, target - statement_unique),
                "ready": statement_unique >= target,
            }
        )
    return out


def _paper_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = {
        "benchmark",
        "condition",
        "seed_split",
        "candidate_count",
        "retained_count",
        "statement_unique_count",
        "near_copy_candidate_count",
        "near_copy_candidate_rate_pct",
        "mean_parent_similarity",
        "parent_answer_same_count",
        "parent_answer_same_rate_pct",
        "code_execution_error_count",
        "code_execution_error_rate_pct",
        "answer_mismatch_count",
        "answer_mismatch_rate_pct",
        "shadow_solvability_checked_count",
        "shadow_solvability_fail_count",
        "shadow_solvability_fail_rate_pct",
        "critical_error_proxy_count",
        "critical_error_proxy_rate_pct",
    }
    return [{key: row.get(key, "") for key in row if key in keep} for row in _aggregate(rows, ["benchmark", "condition", "seed_split"])]


def _write_markdown_summary(path: Path, condition_counts: List[Dict[str, Any]], paper_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Expanded Ablation Microstudy Summary",
        "",
        "This summary is derived from external-ablation run artifacts and is intended for paper/supplementary consistency checks.",
        "",
        "## Cell Readiness",
        "",
        "| benchmark | condition | candidates | statement-unique | target | ready |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in condition_counts:
        lines.append(
            "| {benchmark} | {condition} | {candidate_count} | {statement_unique_count} | {target_statement_unique_candidates} | {ready} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Paper Table",
            "",
            "| benchmark | condition | split | N | unique | near-copy % | parent-same % | code err % | answer mismatch % | shadow fail % | critical % |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paper_rows:
        lines.append(
            "| {benchmark} | {condition} | {seed_split} | {candidate_count} | {statement_unique_count} | {near_copy_candidate_rate_pct} | {parent_answer_same_rate_pct} | {code_execution_error_rate_pct} | {answer_mismatch_rate_pct} | {shadow_solvability_fail_rate_pct} | {critical_error_proxy_rate_pct} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--run-dirs", nargs="+")
    parser.add_argument("--discover-external-ablation-runs", action="store_true")
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--min-statement-unique-per-cell", type=int, default=0)
    parser.add_argument("--run-shadow-solvability", action="store_true")
    parser.add_argument("--skip-code-execution", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    combo_lookup = _manifest_lookup(manifest)
    run_dirs: List[Path] = []
    if args.discover_external_ablation_runs:
        run_dirs.extend(_discover_external_ablation_run_dirs(args.conditions))
    for raw_run_dir in args.run_dirs or []:
        run_dir = Path(raw_run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO / run_dir
        run_dirs.append(run_dir)
    run_dirs = sorted(dict.fromkeys(run_dirs))
    if not run_dirs:
        raise SystemExit("no run dirs provided; pass --run-dirs or --discover-external-ablation-runs")

    rows: List[Dict[str, Any]] = []
    wanted_conditions = {str(condition) for condition in args.conditions or []}
    for run_dir in run_dirs:
        if not (run_dir / "full_run_result.json").exists():
            continue
        summary = _load_json(run_dir / "full_run_result.json")
        labels = _infer_run_labels(summary, combo_lookup)
        if wanted_conditions and labels.get("condition") not in wanted_conditions:
            continue
        rows.extend(
            analyze_run(
                run_dir,
                combo_lookup,
                run_shadow=bool(args.run_shadow_solvability),
                execute_code=not bool(args.skip_code_execution),
            )
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_benchmark = _aggregate(rows, ["benchmark", "condition", "seed_split"])
    condition_counts = _condition_counts(rows, args.min_statement_unique_per_cell)
    paper_rows = _paper_table(rows)
    _write_csv(out_dir / "ablation_candidates.csv", rows)
    _write_csv(out_dir / "ablation_summary.csv", _aggregate(rows, ["condition", "seed_split"]))
    _write_csv(out_dir / "ablation_by_benchmark.csv", by_benchmark)
    _write_csv(out_dir / "ablation_condition_counts.csv", condition_counts)
    _write_csv(out_dir / "ablation_paper_table.csv", paper_rows)
    _write_markdown_summary(out_dir / "ablation_paper_summary.md", condition_counts, paper_rows)

    with (out_dir / "ablation_failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["near_copy_candidate"] or row["code_execution_error"] or row["answer_mismatch"] or row["shadow_solvability_fail"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.min_statement_unique_per_cell:
        underfilled = [row for row in condition_counts if not row["ready"]]
        if underfilled:
            details = ", ".join(
                f"{row['benchmark']}/{row['condition']}={row['statement_unique_count']}/{args.min_statement_unique_per_cell}"
                for row in underfilled
            )
            raise SystemExit(f"ablation target not met: {details}")

    print(
        json.dumps(
            {
                "run_dirs": len(run_dirs),
                "candidate_rows": len(rows),
                "out_dir": str(out_dir),
                "condition_counts": condition_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
