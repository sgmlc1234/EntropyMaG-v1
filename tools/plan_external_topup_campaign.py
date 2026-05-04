#!/usr/bin/env python3
"""Plan top-up generation commands for expanded external treatment targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from pathlib import Path
from typing import Dict, List

from summarize_external_expansion_progress import summarize


REPO = Path(__file__).resolve().parents[1]
BENCHES = ("math500", "aime2025", "gsm8k")
COMBOS = ("new_a", "new_b", "new_c", "new_d")
DEFAULT_OUT = REPO / "data/analysis/external_expansion_progress/topup_campaign_plan.csv"


def _quote(value: object) -> str:
    return shlex.quote(str(value))


def build_plan(target: int, cushion: int, max_generations: int) -> List[Dict[str, object]]:
    _, summary_rows = summarize(target)
    by_bench = {row["benchmark"]: row for row in summary_rows}
    plan_rows: List[Dict[str, object]] = []
    for bench in BENCHES:
        remaining = int(by_bench[bench]["remaining_to_target"])
        planned_total = remaining + (cushion if remaining else 0)
        per_combo = math.ceil(planned_total / len(COMBOS)) if planned_total else 0
        for combo in COMBOS:
            if not per_combo:
                continue
            cmd = [
                "tools/launch_ablation_run.sh",
                bench,
                combo,
                "full",
                str(max_generations),
                str(per_combo),
            ]
            plan_rows.append(
                {
                    "benchmark": bench,
                    "combo": combo,
                    "target_statement_unique_treatments": target,
                    "current_full_total": by_bench[bench]["current_full_total"],
                    "remaining_to_target": remaining,
                    "planned_total_with_cushion": planned_total,
                    "target_problem_count_per_combo": per_combo,
                    "max_generations": max_generations,
                    "command": " ".join(_quote(part) for part in cmd),
                }
            )
    return plan_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=250)
    parser.add_argument("--cushion", type=int, default=20, help="Extra planned rows to absorb dedup/failed runs.")
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = build_plan(args.target, args.cushion, args.max_generations)
    out = args.out if args.out.is_absolute() else REPO / args.out
    write_csv(out, rows)
    print(json.dumps({"jobs": len(rows), "out": str(out.relative_to(REPO)), "plan": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
