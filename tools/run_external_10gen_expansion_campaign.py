#!/usr/bin/env python3
"""Run target-free 10-generation external expansion jobs until treatment pools are ready.

This campaign runner intentionally does not pass --target-problem-count to the
generation pipeline. Each launched job is a full max-generation run, so the
resulting artifacts preserve generation_1.json ... generation_N.json for later
generation-wise evaluation.
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

from summarize_external_expansion_progress import summarize  # noqa: E402


BENCHES = ("math500", "aime2025", "gsm8k")
COMBOS = ("new_a", "new_b", "new_c", "new_d")
DEFAULT_OUT_DIR = REPO / "data/analysis/external_expansion_progress"


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"attempts": [], "combo_cursor": {bench: 0 for bench in BENCHES}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"attempts": [], "combo_cursor": {bench: 0 for bench in BENCHES}}
    state.setdefault("attempts", [])
    state.setdefault("combo_cursor", {bench: 0 for bench in BENCHES})
    for bench in BENCHES:
        state["combo_cursor"].setdefault(bench, 0)
    return state


def _write_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_quiet(cmd: List[str], log_handle) -> subprocess.CompletedProcess[str]:
    log_handle.write(f"[campaign] $ {' '.join(cmd)}\n")
    log_handle.flush()
    result = subprocess.run(
        cmd,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_handle.write(result.stdout)
    if result.stdout and not result.stdout.endswith("\n"):
        log_handle.write("\n")
    log_handle.write(f"[campaign] command rc={result.returncode}\n")
    log_handle.flush()
    return result


def rebuild_and_summarize(target: int, log_handle) -> List[Dict[str, Any]]:
    _run_quiet(
        [
            "python3",
            "tools/build_external_ablation_eval_datasets.py",
            "--auto-new-full-treatment",
            "--build-expanded-full",
            "--expanded-target",
            str(target),
            "--allow-underfilled-expanded",
        ],
        log_handle,
    )
    _run_quiet(
        [
            "python3",
            "tools/summarize_external_expansion_progress.py",
            "--target",
            str(target),
        ],
        log_handle,
    )
    _, rows = summarize(target)
    return rows


def _summary_by_bench(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["benchmark"]): row for row in rows}


def _remaining(row: Dict[str, Any]) -> int:
    return int(row.get("remaining_to_target", 0) or 0)


def _count_attempts(state: Dict[str, Any], bench: str, combo: Optional[str] = None) -> int:
    out = 0
    for attempt in state.get("attempts", []):
        if attempt.get("benchmark") != bench:
            continue
        if combo is not None and attempt.get("combo") != combo:
            continue
        out += 1
    return out


def choose_next_job(
    rows: List[Dict[str, Any]],
    state: Dict[str, Any],
    allowed_benches: List[str],
    allowed_combos: List[str],
) -> Optional[tuple[str, str]]:
    underfilled = [row for row in rows if row["benchmark"] in allowed_benches and _remaining(row) > 0]
    if not underfilled:
        return None

    # Primary priority is remaining deficit. The attempt count tiebreaker keeps
    # the campaign from hammering a single benchmark forever when deficits are close.
    underfilled.sort(
        key=lambda row: (
            -_remaining(row),
            _count_attempts(state, str(row["benchmark"])),
            str(row["benchmark"]),
        )
    )
    bench = str(underfilled[0]["benchmark"])

    combo_counts = [(combo, _count_attempts(state, bench, combo)) for combo in allowed_combos]
    min_count = min(count for _, count in combo_counts)
    candidates = [combo for combo, count in combo_counts if count == min_count]
    cursor = int(state.get("combo_cursor", {}).get(bench, 0) or 0)
    for offset in range(len(allowed_combos)):
        combo = allowed_combos[(cursor + offset) % len(allowed_combos)]
        if combo in candidates:
            state["combo_cursor"][bench] = (allowed_combos.index(combo) + 1) % len(allowed_combos)
            return bench, combo
    return bench, candidates[0]


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


def run_generation_job(
    bench: str,
    combo: str,
    max_generations: int,
    idle_timeout: int,
    log_handle,
) -> Dict[str, Any]:
    cmd = f"tools/launch_ablation_run.sh {bench} {combo} full {max_generations}"
    log_handle.write(
        f"\n[campaign] starting bench={bench} combo={combo} condition=full max_generations={max_generations}\n"
    )
    log_handle.write(f"[campaign] command={cmd}\n")
    log_handle.flush()
    before = _run_dirs()
    proc = subprocess.Popen(
        ["bash", "-lc", f"set -a; source .env; set +a; {cmd}"],
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
            msg = (
                f"\n[campaign] idle timeout after {idle_timeout}s; "
                f"terminating bench={bench} combo={combo}\n"
            )
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
    elapsed = round(time.time() - start, 1)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "benchmark": bench,
        "combo": combo,
        "condition": "full",
        "max_generations": max_generations,
        "command": cmd,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "run_dir": run_dir,
        "validated_problem_files": _validated_count(run_dir),
    }
    log_handle.write(f"[campaign] finished {json.dumps(record, ensure_ascii=False)}\n")
    log_handle.flush()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=250)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--idle-timeout", type=int, default=900)
    parser.add_argument("--max-jobs", type=int, default=999)
    parser.add_argument("--bench", choices=BENCHES, action="append")
    parser.add_argument("--combo", choices=COMBOS, action="append")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--state", type=Path, default=DEFAULT_OUT_DIR / "external_10gen_campaign_state.json")
    parser.add_argument("--log", type=Path, default=DEFAULT_OUT_DIR / "external_10gen_campaign.log")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state if args.state.is_absolute() else REPO / args.state
    log_path = args.log if args.log.is_absolute() else REPO / args.log
    state = _load_state(state_path)
    allowed_benches = args.bench or list(BENCHES)
    allowed_combos = args.combo or list(COMBOS)

    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n[campaign] session start {datetime.now().isoformat(timespec='seconds')} "
            f"target={args.target} max_generations={args.max_generations} "
            f"benches={allowed_benches} combos={allowed_combos}\n"
        )
        rows = rebuild_and_summarize(args.target, log_handle)
        for job_idx in range(1, args.max_jobs + 1):
            before_rows = rows
            before_by_bench = _summary_by_bench(before_rows)
            next_job = choose_next_job(rows, state, allowed_benches, allowed_combos)
            if next_job is None:
                log_handle.write("[campaign] all requested benchmark targets are met; stopping.\n")
                break
            bench, combo = next_job
            if args.dry_run:
                print(f"dry-run job {job_idx}: tools/launch_ablation_run.sh {bench} {combo} full {args.max_generations}")
                state["attempts"].append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "benchmark": bench,
                        "combo": combo,
                        "condition": "full",
                        "max_generations": args.max_generations,
                        "dry_run": True,
                    }
                )
                _write_state(state_path, state)
                rows = before_rows
                continue

            record = run_generation_job(bench, combo, args.max_generations, args.idle_timeout, log_handle)
            rows = rebuild_and_summarize(args.target, log_handle)
            after_by_bench = _summary_by_bench(rows)
            before_total = int(before_by_bench.get(bench, {}).get("current_full_total", 0) or 0)
            after_total = int(after_by_bench.get(bench, {}).get("current_full_total", 0) or 0)
            record["quality_gated_pool_before"] = before_total
            record["quality_gated_pool_after"] = after_total
            record["quality_gated_pool_delta"] = after_total - before_total
            state["attempts"].append(record)
            state["last_summary"] = rows
            _write_state(state_path, state)
            print(
                f"[campaign] progress {bench}: {before_total} -> {after_total} "
                f"(delta {after_total - before_total}); remaining={after_by_bench[bench]['remaining_to_target']}",
                flush=True,
            )
            if all(_remaining(row) == 0 for row in rows if row["benchmark"] in allowed_benches):
                log_handle.write("[campaign] all requested benchmark targets are met after rebuild; stopping.\n")
                break

        rows = rebuild_and_summarize(args.target, log_handle)
        state["last_summary"] = rows
        state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_state(state_path, state)
        print(
            json.dumps(
                {
                    "state": _display(state_path),
                    "log": _display(log_path),
                    "summary": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
