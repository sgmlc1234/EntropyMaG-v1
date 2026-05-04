#!/usr/bin/env python3
"""Run direct no-tool evaluation commands from one or more CSV job matrices."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO / "data/eval_results/eval_job_matrix_runner.log"
DEFAULT_STATE = REPO / "data/eval_results/eval_job_matrix_runner_state.json"


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"attempts": []}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"attempts": []}
    state.setdefault("attempts", [])
    return state


def _write_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_matrix(path: Path) -> List[Dict[str, str]]:
    matrix = path if path.is_absolute() else REPO / path
    with matrix.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for idx, row in enumerate(rows):
        row["_matrix"] = _display(matrix)
        row["_matrix_row_index"] = str(idx)
    return rows


def _output_dir(row: Dict[str, str]) -> Path:
    out = Path(row.get("output_dir", ""))
    return out if out.is_absolute() else REPO / out


def _expected_runs(row: Dict[str, str]) -> int:
    try:
        return int(row.get("dataset_rows", "0") or 0) * int(row.get("repeats", "1") or 1)
    except ValueError:
        return 0


def _completed_runs(row: Dict[str, str]) -> int:
    out = _output_dir(row)
    if not out.is_dir():
        return 0
    return sum(1 for _ in out.glob("*_run_*.json"))


def _summary_complete(row: Dict[str, str]) -> bool:
    expected = _expected_runs(row)
    if expected <= 0:
        return False
    summary = _output_dir(row) / "summary.json"
    if summary.exists():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            if int(payload.get("completed_runs", 0) or 0) >= expected:
                return True
        except Exception:
            pass
    return _completed_runs(row) >= expected


def _row_key(row: Dict[str, str]) -> str:
    parts = [
        row.get("_matrix", ""),
        row.get("_matrix_row_index", ""),
        row.get("benchmark", ""),
        row.get("generation", ""),
        row.get("arm", ""),
        row.get("model", ""),
        row.get("dataset_jsonl", ""),
    ]
    return "|".join(parts)


def _attempts_for(state: Dict[str, Any], key: str) -> int:
    return sum(1 for attempt in state.get("attempts", []) if attempt.get("row_key") == key)


def _run_command(command: str, timeout: int, log_handle) -> Dict[str, Any]:
    log_handle.write(f"\n[eval-matrix] $ {command}\n")
    log_handle.flush()
    proc = subprocess.Popen(
        ["bash", "-lc", f"set -a; source .env; set +a; {command}"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    start = time.time()
    timed_out = False
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if line:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
            continue
        if proc.poll() is not None:
            break
        if timeout and time.time() - start > timeout:
            timed_out = True
            msg = f"\n[eval-matrix] timeout after {timeout}s; terminating command\n"
            print(msg, flush=True)
            log_handle.write(msg)
            log_handle.flush()
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
            break
        time.sleep(1)
    return {
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.time() - start, 1),
    }


def _matrix_progress(rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    rows = list(rows)
    expected = sum(_expected_runs(row) for row in rows)
    completed = sum(min(_completed_runs(row), _expected_runs(row)) for row in rows)
    complete_jobs = sum(1 for row in rows if _summary_complete(row))
    return {
        "jobs": len(rows),
        "complete_jobs": complete_jobs,
        "expected_runs": expected,
        "completed_runs": completed,
        "remaining_runs": max(0, expected - completed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, action="append", required=True)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means no explicit job limit.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=0, help="Per-command timeout in seconds; 0 disables.")
    parser.add_argument("--skip-not-ready", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log_path = args.log if args.log.is_absolute() else REPO / args.log
    state_path = args.state if args.state.is_absolute() else REPO / args.state
    state = _load_state(state_path)
    rows: List[Dict[str, str]] = []
    for matrix in args.matrix:
        rows.extend(_read_matrix(matrix))
    if args.skip_not_ready:
        rows = [row for row in rows if row.get("ready", "true").lower() == "true"]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("OPENROUTER_API_KEY"):
        # The command itself sources .env, but this check catches shell sessions
        # where the runner was launched without env and .env is missing.
        env_path = REPO / ".env"
        if not env_path.exists() or "OPENROUTER_API_KEY" not in env_path.read_text(encoding="utf-8", errors="ignore"):
            raise SystemExit("OPENROUTER_API_KEY is not available in environment or .env.")

    ran_jobs = 0
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n[eval-matrix] session start {datetime.now().isoformat(timespec='seconds')} "
            f"matrices={[str(m) for m in args.matrix]} rows={len(rows)} dry_run={args.dry_run}\n"
        )
        for row in rows:
            if args.max_jobs and ran_jobs >= args.max_jobs:
                break
            key = _row_key(row)
            expected = _expected_runs(row)
            completed_before = _completed_runs(row)
            if _summary_complete(row):
                continue
            if _attempts_for(state, key) > args.retries:
                continue

            command = row.get("command", "")
            if not command:
                continue
            if args.dry_run:
                print(f"dry-run: {command}")
                record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "row_key": key,
                    "dry_run": True,
                    "expected_runs": expected,
                    "completed_before": completed_before,
                    **{k: row.get(k, "") for k in ("_matrix", "_matrix_row_index", "benchmark", "generation", "arm", "model")},
                }
            else:
                attempts_used = 0
                result: Dict[str, Any] = {"returncode": 1, "timed_out": False, "elapsed_seconds": 0}
                completed_after = completed_before
                for attempt_no in range(args.retries + 1):
                    attempts_used = attempt_no + 1
                    result = _run_command(command, args.timeout, log_handle)
                    completed_after = _completed_runs(row)
                    if result.get("returncode") == 0 or _summary_complete(row):
                        break
                    if attempt_no < args.retries:
                        msg = (
                            f"[eval-matrix] retrying after rc={result.get('returncode')} "
                            f"in {args.retry_sleep}s ({attempt_no + 1}/{args.retries})\n"
                        )
                        print(msg, end="", flush=True)
                        log_handle.write(msg)
                        log_handle.flush()
                        time.sleep(args.retry_sleep)
                record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "row_key": key,
                    "expected_runs": expected,
                    "completed_before": completed_before,
                    "completed_after": completed_after,
                    "summary_complete": _summary_complete(row),
                    "attempts_used": attempts_used,
                    **result,
                    **{k: row.get(k, "") for k in ("_matrix", "_matrix_row_index", "benchmark", "generation", "arm", "model")},
                }
                print(
                    "[eval-matrix] progress "
                    f"{row.get('benchmark', '')} "
                    f"{row.get('generation') or row.get('arm', '')} "
                    f"{row.get('model', '')}: {completed_before}->{completed_after}/{expected} "
                    f"rc={result['returncode']}",
                    flush=True,
                )
            state.setdefault("attempts", []).append(record)
            state["last_progress"] = _matrix_progress(rows)
            _write_state(state_path, state)
            ran_jobs += 1

        state["last_progress"] = _matrix_progress(rows)
        state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_state(state_path, state)
        print(
            json.dumps(
                {
                    "state": _display(state_path),
                    "log": _display(log_path),
                    "ran_jobs": ran_jobs,
                    "progress": state["last_progress"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
