#!/usr/bin/env python3
"""Run expanded no_near_copy/no_solvability ablation top-up jobs.

This runner consumes the same planning logic as
plan_ablation_condition_topup_campaign.py, runs one planned job at a time, and
refreshes the plan after each job so the campaign can stop as soon as all
benchmark-condition targets are met.
"""

from __future__ import annotations

import argparse
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
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from plan_ablation_condition_topup_campaign import (  # noqa: E402
    BENCHES,
    CONDITIONS,
    COMBOS,
    build_count_rows,
    build_plan,
    write_csv,
)


DEFAULT_OUT_DIR = REPO / "data/analysis/external_expansion_progress"


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


def _run_dirs() -> set[Path]:
    root = REPO / "data/runs"
    if not root.is_dir():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir()}


def _detect_new_run_dir(before: set[Path]) -> str:
    after = _run_dirs()
    new_dirs = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    if new_dirs:
        return _display(new_dirs[0])
    existing = sorted(after, key=lambda path: path.stat().st_mtime, reverse=True)
    return _display(existing[0]) if existing else ""


def _validated_count(run_dir: str) -> int:
    if not run_dir:
        return 0
    path = REPO / run_dir if not Path(run_dir).is_absolute() else Path(run_dir)
    vp = path / "validated_problems"
    if not vp.is_dir():
        return 0
    return sum(1 for _ in vp.glob("*.json"))


def _filtered_rows(
    rows: Iterable[Dict[str, object]],
    benches: Optional[List[str]],
    conditions: Optional[List[str]],
    combos: Optional[List[str]],
) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        if benches and row["benchmark"] not in benches:
            continue
        if conditions and row["condition"] not in conditions:
            continue
        if combos and row["combo"] not in combos:
            continue
        out.append(row)
    return out


def _attempt_count(state: Dict[str, Any], row: Dict[str, object]) -> int:
    out = 0
    for attempt in state.get("attempts", []):
        if attempt.get("benchmark") != row.get("benchmark"):
            continue
        if attempt.get("condition") != row.get("condition"):
            continue
        if attempt.get("combo") != row.get("combo"):
            continue
        out += 1
    return out


def _choose_next(rows: List[Dict[str, object]], state: Dict[str, Any]) -> Optional[Dict[str, object]]:
    if not rows:
        return None
    rows = sorted(
        rows,
        key=lambda row: (
            -int(row["remaining_to_target"]),
            _attempt_count(state, row),
            str(row["benchmark"]),
            str(row["condition"]),
            str(row["combo"]),
        ),
    )
    return rows[0]


def write_plan_files(
    target: int,
    cushion: int,
    max_generations: int,
    out_path: Path,
) -> List[Dict[str, object]]:
    rows = build_plan(target, cushion, max_generations)
    count_rows = build_count_rows(target)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, rows)
    write_csv(out_path.with_name("ablation_condition_counts.csv"), count_rows)
    return rows


def run_job(command: str, idle_timeout: int, log_handle) -> Dict[str, Any]:
    log_handle.write(f"\n[ablation-campaign] starting command={command}\n")
    log_handle.flush()
    before = _run_dirs()
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
    last_output = time.time()
    timed_out = False
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if line:
            last_output = time.time()
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
            continue
        if proc.poll() is not None:
            break
        if time.time() - last_output > idle_timeout:
            timed_out = True
            msg = f"\n[ablation-campaign] idle timeout after {idle_timeout}s; terminating command={command}\n"
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

    run_dir = _detect_new_run_dir(before)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.time() - start, 1),
        "run_dir": run_dir,
        "validated_problem_files": _validated_count(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--cushion", type=int, default=10)
    parser.add_argument("--max-generations", type=int, default=6)
    parser.add_argument("--idle-timeout", type=int, default=900)
    parser.add_argument("--max-jobs", type=int, default=36)
    parser.add_argument("--bench", choices=BENCHES, action="append")
    parser.add_argument("--condition", choices=CONDITIONS, action="append")
    parser.add_argument("--combo", choices=COMBOS, action="append")
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_OUT_DIR / "ablation_condition_topup_plan.csv",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_OUT_DIR / "ablation_condition_topup_state.json",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_OUT_DIR / "ablation_condition_topup.log",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else REPO / args.plan
    state_path = args.state if args.state.is_absolute() else REPO / args.state
    log_path = args.log if args.log.is_absolute() else REPO / args.log
    state = _load_state(state_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n[ablation-campaign] session start {datetime.now().isoformat(timespec='seconds')} "
            f"target={args.target} max_generations={args.max_generations} "
            f"benches={args.bench or list(BENCHES)} conditions={args.condition or list(CONDITIONS)} "
            f"combos={args.combo or list(COMBOS)}\n"
        )
        for job_idx in range(1, args.max_jobs + 1):
            rows = write_plan_files(args.target, args.cushion, args.max_generations, plan_path)
            rows = _filtered_rows(rows, args.bench, args.condition, args.combo)
            row = _choose_next(rows, state)
            if row is None:
                log_handle.write("[ablation-campaign] all requested targets are met; stopping.\n")
                break

            command = str(row["command"])
            if args.dry_run:
                print(f"dry-run job {job_idx}: {command}")
                state["attempts"].append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "dry_run": True,
                        **row,
                    }
                )
                _write_state(state_path, state)
                continue

            record = {
                "benchmark": row["benchmark"],
                "condition": row["condition"],
                "combo": row["combo"],
                "target_statement_unique_candidates": row["target_statement_unique_candidates"],
                "current_statement_unique_candidates_before": row["current_statement_unique_candidates"],
                "remaining_to_target_before": row["remaining_to_target"],
                **run_job(command, args.idle_timeout, log_handle),
            }
            rows_after = write_plan_files(args.target, args.cushion, args.max_generations, plan_path)
            for after in rows_after:
                if (
                    after["benchmark"] == record["benchmark"]
                    and after["condition"] == record["condition"]
                ):
                    record["current_statement_unique_candidates_after"] = after[
                        "current_statement_unique_candidates"
                    ]
                    record["remaining_to_target_after"] = after["remaining_to_target"]
                    break
            state["attempts"].append(record)
            state["last_counts"] = build_count_rows(args.target)
            _write_state(state_path, state)
            print(
                "[ablation-campaign] progress "
                f"{record['benchmark']} {record['condition']}: "
                f"{record.get('current_statement_unique_candidates_before')} -> "
                f"{record.get('current_statement_unique_candidates_after')} "
                f"(remaining {record.get('remaining_to_target_after')})",
                flush=True,
            )

        rows = write_plan_files(args.target, args.cushion, args.max_generations, plan_path)
        state["last_counts"] = build_count_rows(args.target)
        state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_state(state_path, state)
        print(
            json.dumps(
                {
                    "state": _display(state_path),
                    "log": _display(log_path),
                    "plan": _display(plan_path),
                    "remaining_jobs": len(_filtered_rows(rows, args.bench, args.condition, args.combo)),
                    "counts": state["last_counts"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
