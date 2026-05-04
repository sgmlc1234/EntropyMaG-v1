#!/usr/bin/env python3
"""Apply a conservative quality gate before packaging EntropyMath release rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE = Path("data/public/validated_deep_runs_csv/entropymath_validated_deep_runs_complete.csv")
DEFAULT_OUTPUT_DIR = Path("data/public/quality_gate")
REQUIRED_COLUMNS = ["id", "statement", "answer", "solution", "verification_code"]
QUALITY_COLUMNS = [
    "release_id",
    "quality_status",
    "quality_severity",
    "quality_reasons",
    "quality_runtime_status",
    "quality_stdout_last_line",
    "quality_stderr_excerpt",
]


HARD_TEXT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "solution_provided_answer_incorrect",
        re.compile(r"\bprovided\s+(?:final\s+)?answer\b.{0,120}\bincorrect\b", re.I | re.S),
    ),
    (
        "solution_candidate_answer_incorrect",
        re.compile(r"\bcandidate(?:\s+answer)?\b.{0,120}\bincorrect\b", re.I | re.S),
    ),
    (
        "solution_final_answer_incorrect",
        re.compile(r"\bfinal\s+answer\b.{0,120}\bincorrect\b", re.I | re.S),
    ),
    (
        "solution_given_answer_incorrect",
        re.compile(r"\bgiven\s+answer\b.{0,120}\bincorrect\b", re.I | re.S),
    ),
    (
        "solution_answer_contradiction",
        re.compile(
            r"\b(?:correct|actual|true)\s+(?:sum|calculation|answer|result|value)\s+(?:is|should\s+be|equals|yields)\b",
            re.I,
        ),
    ),
    (
        "solution_verification_code_incorrect",
        re.compile(r"\bverification\s+code\b.{0,120}\bincorrect\b", re.I | re.S),
    ),
]


SUPPORT_TEXT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("support_sandbox_evidence", re.compile(r"\bsandbox\s+evidence\b", re.I)),
    ("support_sandbox_output", re.compile(r"\bsandbox\s+output\b", re.I)),
    ("support_tool_evidence", re.compile(r"\btool\s+evidence\b", re.I)),
    ("support_based_on_sandbox", re.compile(r"\bbased\s+on\s+(?:the\s+)?sandbox\b", re.I)),
    ("support_provided_tool_evidence", re.compile(r"\bprovided\s+tool\s+evidence\b", re.I)),
    ("support_wait_recheck", re.compile(r"\bwait[,;:]?\s+", re.I)),
    ("support_re_evaluating", re.compile(r"\bre[- ]?evaluat(?:e|ing|ion)\b", re.I)),
    ("support_re_checking", re.compile(r"\bre[- ]?check(?:ing|ed)?\b", re.I)),
    ("support_re_calculating", re.compile(r"\bre[- ]?calculat(?:e|ing|ion)\b", re.I)),
    ("support_appears", re.compile(r"\bappears\s+(?:to|that)\b", re.I)),
    ("support_suggests", re.compile(r"\bsuggests?\b", re.I)),
    ("support_implies_different", re.compile(r"\bimplies?\s+a\s+different\b", re.I)),
]


CODE_SUPPORT_TERMS = [
    "known to be",
    "provided verification code",
    "target answer",
    "sandbox evidence",
    "sandbox output",
    "tool evidence",
    "hardcoded",
    "hard-coded",
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


def release_id(row: dict[str, str]) -> str:
    statement_hash = row.get("statement_sha256") or hashlib.sha256(row.get("statement", "").encode("utf-8")).hexdigest()
    return f"emv1_{statement_hash[:16]}"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return columns, rows


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def display_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def sanitize_runtime_text(text: str) -> str:
    sanitized = str(text or "")
    sanitized = re.sub(r'File ".*?/check\.py"', 'File "<verification_code>"', sanitized)
    sanitized = re.sub(r"/Users/[^/\s]+/[^\s\"']*", "<local-path>", sanitized)
    sanitized = re.sub(r"/var/folders/[^\s\"']*", "<temp-path>", sanitized)
    return sanitized


def run_verification(code: str, python_exe: str, timeout_s: int) -> tuple[str, str, str]:
    if not str(code or "").strip():
        return "missing", "", "empty verification_code"
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "check.py"
        script.write_text(str(code) + "\n", encoding="utf-8")
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
    stderr = sanitize_runtime_text((proc.stderr or "").strip())
    return status, stdout, stderr[:1000]


def stdout_agrees(stdout: str, answer: str) -> bool:
    lines = [line.strip() for line in str(stdout or "").splitlines() if line.strip()]
    last_line = lines[-1] if lines else ""
    return (
        norm_text(last_line) == norm_text(answer)
        or numeric_equivalent(last_line, answer)
        or numeric_equivalent(last_numeric_token(last_line), answer)
    )


def hardcoded_answer_candidate(code: str, answer: str) -> bool:
    answer_norm = norm_text(answer)
    if not answer_norm or len(answer_norm) > 64:
        return False
    compact = norm_text(code)
    escaped = re.escape(answer_norm)
    direct_literal = bool(re.search(rf"(?:return|print)\(?['\"]?{escaped}['\"]?\)?", compact))
    return direct_literal


def quality_reasons_for_row(
    row: dict[str, str],
    python_exe: str,
    timeout_s: int,
    skip_runtime: bool,
    skip_static_excluded_runtime: bool,
) -> tuple[str, str, list[str], str, str, str]:
    solution = row.get("solution", "")
    code = row.get("verification_code", "")
    answer = row.get("answer", "")
    hard_reasons: list[str] = []
    support_reasons: list[str] = []

    for name, pattern in HARD_TEXT_PATTERNS:
        if pattern.search(solution):
            hard_reasons.append(name)
    for name, pattern in SUPPORT_TEXT_PATTERNS:
        if pattern.search(solution):
            support_reasons.append(name)
    lower_code = code.lower()
    for term in CODE_SUPPORT_TERMS:
        if term in lower_code:
            support_reasons.append("support_code_" + term.replace(" ", "_").replace("-", "_"))
    if hardcoded_answer_candidate(code, answer):
        support_reasons.append("support_code_direct_answer_literal")

    runtime_status = "skipped"
    stdout_last = ""
    stderr_excerpt = ""
    static_excluded = bool(hard_reasons or support_reasons)
    if skip_static_excluded_runtime and static_excluded:
        runtime_status = "not_run_static_exclude"
    elif not skip_runtime:
        runtime_status, stdout, stderr_excerpt = run_verification(code, python_exe, timeout_s)
        stdout_lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        stdout_last = stdout_lines[-1] if stdout_lines else ""
        if runtime_status != "pass":
            hard_reasons.append(f"verification_code_runtime_{runtime_status}")
        elif not stdout_agrees(stdout, answer):
            hard_reasons.append("verification_stdout_answer_disagreement")

    if hard_reasons:
        return "excluded", "hard_exclude", sorted(set(hard_reasons + support_reasons)), runtime_status, stdout_last, stderr_excerpt
    if support_reasons:
        return "excluded", "support_gap_exclude", sorted(set(support_reasons)), runtime_status, stdout_last, stderr_excerpt
    return "kept", "kept", ["no_quality_gate_flag"], runtime_status, stdout_last, stderr_excerpt


def markdown_summary(summary: dict[str, object]) -> str:
    reason_counts = summary.get("exclusion_reason_counts", {})
    lines = [
        "# EntropyMath quality-gate summary",
        "",
        "This pre-release gate conservatively quarantines rows with explicit answer contradictions, verification-code failures or answer mismatches, and support-gap cues that could weaken reviewer trust in the public release.",
        "",
        f"- Source CSV: `{summary['source_csv']}`",
        f"- Source rows: {summary['source_row_count']:,}",
        f"- Kept rows before statement-dedup packaging: {summary['kept_row_count']:,}",
        f"- Excluded rows: {summary['excluded_row_count']:,}",
        f"- Hard exclusions: {summary['hard_exclude_count']:,}",
        f"- Support-gap exclusions: {summary['support_gap_exclude_count']:,}",
        "",
        "| Reason | Rows |",
        "|---|---:|",
    ]
    for reason, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), item[0])):
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Clean source CSV: `{summary['clean_csv']}`",
            f"- Quarantine CSV: `{summary['quarantine_csv']}`",
            f"- Quarantine JSONL: `{summary['quarantine_jsonl']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, object]:
    source = args.source_csv.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    columns, rows = read_rows(source)

    clean_rows: list[dict[str, str]] = []
    quarantine_rows: list[dict[str, str]] = []
    severity_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    runtime_counts: Counter[str] = Counter()

    for idx, row in enumerate(rows, start=1):
        if args.max_rows and idx > args.max_rows:
            break
        status, severity, reasons, runtime_status, stdout_last, stderr_excerpt = quality_reasons_for_row(
            row,
            args.python,
            args.timeout_s,
            args.skip_runtime,
            args.skip_static_excluded_runtime,
        )
        runtime_counts[runtime_status] += 1
        severity_counts[severity] += 1
        for reason in reasons:
            if reason != "no_quality_gate_flag":
                reason_counts[reason] += 1
        if status == "kept":
            clean_rows.append(dict(row))
        else:
            quarantine = dict(row)
            quarantine.update(
                {
                    "release_id": release_id(row),
                    "quality_status": status,
                    "quality_severity": severity,
                    "quality_reasons": "; ".join(reasons),
                    "quality_runtime_status": runtime_status,
                    "quality_stdout_last_line": stdout_last,
                    "quality_stderr_excerpt": stderr_excerpt,
                }
            )
            quarantine_rows.append(quarantine)
        if args.progress_every and idx % args.progress_every == 0:
            print(
                f"quality_gate_progress processed={idx} kept={len(clean_rows)} excluded={len(quarantine_rows)}",
                file=sys.stderr,
                flush=True,
            )

    clean_csv = out / "entropymath_quality_clean.csv"
    quarantine_csv = out / "entropymath_quality_quarantine.csv"
    quarantine_jsonl = out / "entropymath_quality_quarantine.jsonl"
    summary_json = out / "quality_summary.json"
    summary_md = out / "quality_summary.md"

    write_csv(clean_csv, clean_rows, columns)
    write_csv(quarantine_csv, quarantine_rows, QUALITY_COLUMNS + columns)
    write_jsonl(quarantine_jsonl, quarantine_rows)

    processed_count = len(clean_rows) + len(quarantine_rows)
    summary: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": display_path(source, Path.cwd()),
        "source_row_count": len(rows),
        "processed_row_count": processed_count,
        "kept_row_count": len(clean_rows),
        "excluded_row_count": len(quarantine_rows),
        "hard_exclude_count": severity_counts["hard_exclude"],
        "support_gap_exclude_count": severity_counts["support_gap_exclude"],
        "runtime_status_counts": dict(sorted(runtime_counts.items())),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "clean_csv": display_path(clean_csv, Path.cwd()),
        "quarantine_csv": display_path(quarantine_csv, Path.cwd()),
        "quarantine_jsonl": display_path(quarantine_jsonl, Path.cwd()),
        "summary_md": display_path(summary_md, Path.cwd()),
        "skip_runtime": bool(args.skip_runtime),
        "skip_static_excluded_runtime": bool(args.skip_static_excluded_runtime),
        "timeout_s": args.timeout_s,
        "python": args.python,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(markdown_summary(summary), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-s", type=int, default=10)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-runtime", action="store_true", help="Only apply text/code static quality checks.")
    parser.add_argument(
        "--check-static-excluded",
        dest="skip_static_excluded_runtime",
        action="store_false",
        default=True,
        help="Also run verification_code for rows already excluded by static contradiction/support-gap checks.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Debug-only row limit; 0 means all rows.")
    parser.add_argument("--progress-every", type=int, default=100)
    print(json.dumps(build(parser.parse_args(list(argv) if argv is not None else None)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
