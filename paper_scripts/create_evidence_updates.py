#!/usr/bin/env python3
"""Create reviewer-facing evidence updates.

Outputs:
- Codex-assisted 30-row precheck from the frozen 180-row audit sample.
- Control-count Wilson confidence intervals for the external benchmark table.

The precheck is intentionally framed as a machine-assisted triage pass, not as
completed human audit evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
AUDIT_SAMPLE = WORKSPACE / "repo/data/public/evaluation_samples/human_audit_stratified_180.csv"
AUDIT_OUT_DIR = ROOT / "analysis/audit_precheck"
EXTERNAL_OUT_DIR = ROOT / "analysis/external_benchmarks"

QUALITY_FLAG_PATTERNS = [
    ("provided_answer_incorrect", re.compile(r"\bprovided\s+(?:final\s+)?answer\b.{0,120}\bincorrect\b", re.I | re.S)),
    ("candidate_answer_incorrect", re.compile(r"\bcandidate(?:\s+answer)?\b.{0,120}\bincorrect\b", re.I | re.S)),
    ("final_answer_incorrect", re.compile(r"\bfinal\s+answer\b.{0,120}\bincorrect\b", re.I | re.S)),
    ("given_answer_incorrect", re.compile(r"\bgiven\s+answer\b.{0,120}\bincorrect\b", re.I | re.S)),
    ("sandbox_evidence", re.compile(r"\bsandbox\s+evidence\b", re.I)),
    ("sandbox_output", re.compile(r"\bsandbox\s+output\b", re.I)),
    ("tool_evidence", re.compile(r"\btool\s+evidence\b", re.I)),
    ("based_on_sandbox", re.compile(r"\bbased\s+on\s+(?:the\s+)?sandbox\b", re.I)),
    ("wait_recheck", re.compile(r"\bwait[,;:]?\s+", re.I)),
    ("re_evaluating", re.compile(r"\bre[- ]?evaluat(?:e|ing|ion)\b", re.I)),
    ("re_checking", re.compile(r"\bre[- ]?check(?:ing|ed)?\b", re.I)),
    ("re_calculating", re.compile(r"\bre[- ]?calculat(?:e|ing|ion)\b", re.I)),
    ("appears", re.compile(r"\bappears\s+(?:to|that)\b", re.I)),
    ("suggests", re.compile(r"\bsuggests?\b", re.I)),
    ("implies_different", re.compile(r"\bimplies?\s+a\s+different\b", re.I)),
]


def norm_text(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = text.strip().strip("$").strip()
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def numeric_equivalent(a: object, b: object, tol: float = 1e-6) -> bool:
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


def last_numeric_token(text: object) -> str:
    tokens = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", str(text or ""))
    return tokens[-1] if tokens else ""


def select_stratified(rows: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("difficulty_label", ""), row.get("operation", ""))].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    ordered_groups = sorted(groups)

    for key in ordered_groups:
        bucket = list(groups[key])
        rng.shuffle(bucket)
        if bucket:
            row = bucket[0]
            selected.append(row)
            selected_ids.add(row["release_id"])

    remaining_budget = max(0, n - len(selected))
    leftovers = [row for row in rows if row.get("release_id") not in selected_ids]
    rng.shuffle(leftovers)

    weights = {
        key: len([row for row in bucket if row.get("release_id") not in selected_ids])
        for key, bucket in groups.items()
    }
    total_weight = sum(weights.values())
    allocations = {key: 0 for key in ordered_groups}
    if total_weight and remaining_budget:
        raw_alloc = {key: remaining_budget * weights[key] / total_weight for key in ordered_groups}
        allocations = {key: int(math.floor(raw_alloc[key])) for key in ordered_groups}
        gap = remaining_budget - sum(allocations.values())
        for key in sorted(ordered_groups, key=lambda k: raw_alloc[k] - allocations[k], reverse=True)[:gap]:
            allocations[key] += 1

    by_group_leftovers: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in leftovers:
        by_group_leftovers[(row.get("difficulty_label", ""), row.get("operation", ""))].append(row)

    for key in ordered_groups:
        for row in by_group_leftovers[key][: allocations[key]]:
            if len(selected) >= n:
                break
            selected.append(row)
            selected_ids.add(row["release_id"])

    if len(selected) < n:
        for row in leftovers:
            if row.get("release_id") in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["release_id"])
            if len(selected) >= n:
                break

    return sorted(selected[:n], key=lambda r: (r.get("difficulty_label", ""), r.get("operation", ""), r["release_id"]))


def run_verification(code: str, python_exe: str, timeout_s: int) -> tuple[str, str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "check.py"
        script.write_text(code + "\n", encoding="utf-8")
        try:
            proc = subprocess.run(
                [python_exe, "-I", str(script)],
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return "timeout", "", "timeout"
    status = "pass" if proc.returncode == 0 else "error"
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    return status, stdout, stderr[:500]


def hardcoded_answer_candidate(code: str, answer: str) -> bool:
    answer_norm = norm_text(answer)
    if not answer_norm:
        return False
    compact = norm_text(code)
    simple_answer = re.escape(answer_norm)
    direct = bool(re.search(rf"(return|print)\(?['\"]?{simple_answer}['\"]?\)?", compact))
    support_gap_terms = ["known to be", "provided verification code", "computational model", "exact sum is"]
    return direct or any(term in code.lower() for term in support_gap_terms)


def solution_mismatch_candidate(solution: str, answer: str) -> bool:
    lower = solution.lower()
    if "provided answer" in lower and "incorrect" in lower:
        return True
    if "final value is" in lower and "as per the provided verification code" in lower:
        return True
    return False


def create_precheck(args: argparse.Namespace) -> dict[str, int]:
    with AUDIT_SAMPLE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_stratified(rows, args.precheck_n, args.seed)

    out_rows: list[dict[str, str]] = []
    for row in selected:
        status, stdout, stderr = run_verification(row.get("verification_code", ""), args.python, args.timeout_s)
        stdout_last = stdout.splitlines()[-1].strip() if stdout.splitlines() else ""
        code_agrees = status == "pass" and (
            norm_text(stdout_last) == norm_text(row.get("answer", ""))
            or numeric_equivalent(stdout_last, row.get("answer", ""))
            or numeric_equivalent(last_numeric_token(stdout_last), row.get("answer", ""))
        )
        solution_mismatch = solution_mismatch_candidate(row.get("solution", ""), row.get("answer", ""))
        hardcoded = hardcoded_answer_candidate(row.get("verification_code", ""), row.get("answer", ""))
        major = status != "pass" or not code_agrees or solution_mismatch
        minor = (not major) and hardcoded
        severity = "major_candidate" if major else "minor_candidate" if minor else "no_flag"
        notes = []
        if status != "pass":
            notes.append(f"verification_code_runtime_{status}")
        if status == "pass" and not code_agrees:
            notes.append("verification_stdout_answer_disagreement")
        if solution_mismatch:
            notes.append("solution_text_mismatch_cue")
        if hardcoded and not major:
            notes.append("verification_code_support_gap_candidate")
        if not notes:
            notes.append("no_precheck_flag")

        out = dict(row)
        out.update(
            {
                "precheck_label_source": "Codex-assisted computational/text precheck; pending human confirmation",
                "precheck_python_status": status,
                "precheck_stdout_last_line": stdout_last,
                "precheck_stderr_excerpt": stderr,
                "precheck_code_answer_agrees": str(code_agrees).lower(),
                "precheck_major_error_candidate": str(major).lower(),
                "precheck_minor_issue_candidate": str(minor).lower(),
                "precheck_severity": severity,
                "precheck_notes": "; ".join(notes),
            }
        )
        out_rows.append(out)

    AUDIT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = AUDIT_OUT_DIR / "codex_precheck_30.csv"
    fieldnames = list(out_rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    severity_counts = Counter(row["precheck_severity"] for row in out_rows)
    summary = {
        "label_source": "Codex-assisted computational/text precheck; pending human confirmation",
        "source_sample": str(AUDIT_SAMPLE.relative_to(WORKSPACE)),
        "seed": args.seed,
        "n": len(out_rows),
        "major_error_candidate_count": severity_counts["major_candidate"],
        "minor_issue_candidate_count": severity_counts["minor_candidate"],
        "no_flag_count": severity_counts["no_flag"],
        "major_error_candidate_rate": severity_counts["major_candidate"] / len(out_rows),
        "minor_issue_candidate_rate": severity_counts["minor_candidate"] / len(out_rows),
        "output_csv": str(out_csv.relative_to(ROOT)),
    }
    (AUDIT_OUT_DIR / "codex_precheck_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Codex-assisted audit precheck",
        "",
        "This is a computational/text triage pass over a deterministic 30-row stratified subset of the frozen 180-row audit sample. It is not a completed human audit.",
        "",
        f"- Source sample: `{summary['source_sample']}`",
        f"- Selection seed: `{summary['seed']}`",
        f"- Rows checked: {summary['n']}",
        f"- Major error candidates: {summary['major_error_candidate_count']}/{summary['n']} ({summary['major_error_candidate_rate']:.1%})",
        f"- Minor issue candidates: {summary['minor_issue_candidate_count']}/{summary['n']} ({summary['minor_issue_candidate_rate']:.1%})",
        f"- No precheck flag: {summary['no_flag_count']}/{summary['n']}",
        "",
        "Major candidates are rows where `verification_code` failed, its final stdout did not match the released answer, or solution text contained an explicit mismatch cue. Minor candidates are rows where the code-answer agreement passed but `verification_code` or solution text showed a support-gap cue that should be checked by a human auditor.",
        "",
    ]
    (AUDIT_OUT_DIR / "codex_precheck_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def create_external_ci() -> None:
    controls = [
        ("MATH-500 Level 4--5", "GPT-5.4-mini", 9, 10),
        ("MATH-500 Level 4--5", "Claude Haiku 4.5", 10, 10),
        ("MATH-500 Level 4--5", "Gemini 3.1 Flash Lite", 10, 10),
        ("AIME 2025", "GPT-5.4-mini", 4, 10),
        ("AIME 2025", "Claude Haiku 4.5", 6, 10),
        ("AIME 2025", "Gemini 3.1 Flash Lite", 6, 10),
        ("GSM8K", "GPT-5.4-mini", 9, 10),
        ("GSM8K", "Claude Haiku 4.5", 9, 10),
        ("GSM8K", "Gemini 3.1 Flash Lite", 10, 10),
    ]
    EXTERNAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_model = []
    for benchmark, model, successes, n in controls:
        low, high = wilson_ci(successes, n)
        per_model.append(
            {
                "benchmark": benchmark,
                "model": model,
                "control_correct": successes,
                "control_n": n,
                "control_accuracy": successes / n,
                "control_accuracy_wilson95_low": low,
                "control_accuracy_wilson95_high": high,
            }
        )
    with (EXTERNAL_OUT_DIR / "external_control_ci.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_model[0].keys()))
        writer.writeheader()
        writer.writerows(per_model)

    pooled = []
    for benchmark in ["MATH-500 Level 4--5", "AIME 2025", "GSM8K"]:
        sub = [row for row in per_model if row["benchmark"] == benchmark]
        successes = sum(int(row["control_correct"]) for row in sub)
        n = sum(int(row["control_n"]) for row in sub)
        low, high = wilson_ci(successes, n)
        pooled.append(
            {
                "benchmark": benchmark,
                "pooled_control_correct": successes,
                "pooled_control_n": n,
                "pooled_control_accuracy": successes / n,
                "pooled_control_accuracy_wilson95_low": low,
                "pooled_control_accuracy_wilson95_high": high,
            }
        )
    with (EXTERNAL_OUT_DIR / "external_control_ci_by_benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pooled[0].keys()))
        writer.writeheader()
        writer.writerows(pooled)

    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    table = [
        "| Benchmark | Control correct / cells | Pooled control @1 | Wilson 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for row in pooled:
        table.append(
            "| {benchmark} | {pooled_control_correct}/{pooled_control_n} | {acc} | [{low}, {high}] |".format(
                benchmark=row["benchmark"],
                pooled_control_correct=row["pooled_control_correct"],
                pooled_control_n=row["pooled_control_n"],
                acc=pct(float(row["pooled_control_accuracy"])),
                low=pct(float(row["pooled_control_accuracy_wilson95_low"])),
                high=pct(float(row["pooled_control_accuracy_wilson95_high"])),
            )
        )
    lines = [
        "# External benchmark control uncertainty",
        "",
        "Wilson 95% intervals are computed from the published aggregate control counts. The pooled rows combine three model-control cells per benchmark (30 cells total), so they are descriptive uncertainty checks, not independent seed-level inference.",
        "",
        "\n".join(table),
        "",
    ]
    (EXTERNAL_OUT_DIR / "external_control_ci_summary.md").write_text("\n".join(lines), encoding="utf-8")


def quality_flags_for_text(text: str) -> list[str]:
    reasons = []
    for name, pattern in QUALITY_FLAG_PATTERNS:
        if pattern.search(text or ""):
            reasons.append(name)
    return reasons


def create_external_quality_flags() -> dict[str, object]:
    EXTERNAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    external_dir = WORKSPACE / "repo/data/eval/external"
    flagged_rows: list[dict[str, object]] = []
    file_summaries: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()

    for path in sorted(external_dir.glob("*.jsonl")):
        row_count = 0
        flagged_count = 0
        file_reason_counts: Counter[str] = Counter()
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row_count += 1
                row = json.loads(line)
                reasons = quality_flags_for_text(str(row.get("solution", "")))
                if not reasons:
                    continue
                flagged_count += 1
                for reason in reasons:
                    reason_counts[reason] += 1
                    file_reason_counts[reason] += 1
                flagged_rows.append(
                    {
                        "file": path.name,
                        "line_no": line_no,
                        "id": row.get("id", ""),
                        "benchmark": row.get("_source_benchmark", ""),
                        "arm": row.get("_arm", ""),
                        "answer": row.get("answer", ""),
                        "quality_flag_reasons": "; ".join(reasons),
                    }
                )
        file_summaries.append(
            {
                "file": path.name,
                "row_count": row_count,
                "flagged_row_count": flagged_count,
                "quality_flag_reason_counts": dict(sorted(file_reason_counts.items())),
            }
        )

    out_csv = EXTERNAL_OUT_DIR / "external_quality_flags.csv"
    fieldnames = ["file", "line_no", "id", "benchmark", "arm", "answer", "quality_flag_reasons"]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flagged_rows)

    summary = {
        "label": "External treatment/control quality-flag manifest",
        "source_dir": str(external_dir.relative_to(WORKSPACE)),
        "file_count": len(file_summaries),
        "total_row_count": sum(int(row["row_count"]) for row in file_summaries),
        "flagged_row_count": len(flagged_rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "files": file_summaries,
        "output_csv": str(out_csv.relative_to(ROOT)),
        "interpretation": (
            "External treatment arms are validator-passing stress-test arms, not independently audited clean benchmark rows. "
            "This manifest records contradiction/support-gap cues for caveated interpretation."
        ),
    }
    out_json = EXTERNAL_OUT_DIR / "external_quality_flags_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table = [
        "| File | Rows | Flagged | Reason counts |",
        "|---|---:|---:|---|",
    ]
    for item in file_summaries:
        reason_text = ", ".join(f"{key}: {value}" for key, value in item["quality_flag_reason_counts"].items())
        table.append(f"| `{item['file']}` | {item['row_count']} | {item['flagged_row_count']} | {reason_text or '-'} |")
    lines = [
        "# External benchmark quality-flag manifest",
        "",
        "External treatment arms are retained as validator-passing stress-test evidence, not independently audited clean benchmark rows.",
        "The flags below identify explicit contradiction/support-gap cues that should bound interpretation of the external results.",
        "",
        f"- Rows scanned: {summary['total_row_count']}",
        f"- Flagged rows: {summary['flagged_row_count']}",
        "",
        "\n".join(table),
        "",
    ]
    (EXTERNAL_OUT_DIR / "external_quality_flags_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    default_python = WORKSPACE / "repo/.venv/bin/python"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precheck-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--timeout-s", type=int, default=20)
    parser.add_argument("--python", default=str(default_python if default_python.exists() else Path(sys.executable)))
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = create_precheck(args)
    create_external_ci()
    external_flags = create_external_quality_flags()
    print(
        json.dumps(
            {
                "audit_precheck": summary,
                "external_ci": "analysis/external_benchmarks/external_control_ci_summary.md",
                "external_quality_flags": external_flags,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
