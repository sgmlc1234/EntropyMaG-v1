"""Materialize validated children from past runs into a seed-schema JSON.

Filters:
  - op_type in {mutation, crossover} only (skip survivor/fallback)
  - All required fields populated (statement, answer, solution, difficulty)
  - lineage depth-1 only (parent_ids ⊆ original seed pool) — avoids deep drift

Output schema matches data/seed/problems.schema.json: capitalized field names.
"""
import argparse, glob, json, os, sys
from typing import Dict, List, Set


def load_original_seed_ids(seed_files: List[str]) -> Set[str]:
    ids = set()
    for f in seed_files:
        for d in json.load(open(f)):
            ids.add(d.get("ID") or d.get("id") or "")
    return {i for i in ids if i}


def is_depth1(problem: Dict, original_seed_ids: Set[str]) -> bool:
    """True iff every parent_id is in the original seed pool (=> depth 1 child)."""
    parents = problem.get("parent_ids") or []
    if not parents:
        return False
    return all(p in original_seed_ids for p in parents)


def has_required_fields(problem: Dict) -> bool:
    for k in ("statement", "answer", "solution"):
        v = problem.get(k)
        if not v or not str(v).strip():
            return False
    return True


def normalize_difficulty(raw) -> str:
    """Map numeric difficulty (1-10 from rescore) → categorical string used by selector.
    Original seed schema uses Easy/Medium/Hard/Super Hard."""
    if raw is None:
        return "Medium"
    if isinstance(raw, str):
        return raw
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return "Medium"
    if x < 3.5: return "Easy"
    if x < 6.5: return "Medium"
    if x < 8.5: return "Hard"
    return "Super Hard"


def to_seed_schema(problem: Dict) -> Dict:
    """Convert internal problem dict → seed-schema entry."""
    return {
        "ID": problem.get("id"),
        "Question": problem.get("statement"),
        "Answer": str(problem.get("answer", "")),
        "Solution": problem.get("solution") or "",
        "Difficulty": normalize_difficulty(problem.get("difficulty")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs-glob", default="data/runs/*-deep-run",
                    help="Glob for run directories to scan")
    ap.add_argument("--seed-file", action="append", required=True,
                    help="Original seed file (used to identify depth-0 IDs). Repeat for multiple.")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--exclude-difficulty", default="",
                    help="Comma-separated difficulties to exclude (e.g. 'Super Hard')")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    original_ids = load_original_seed_ids(args.seed_file)
    print(f"[materialize] original seed pool size: {len(original_ids)}", file=sys.stderr)

    excluded = {d.strip() for d in args.exclude_difficulty.split(",") if d.strip()}

    by_id: Dict[str, Dict] = {}  # de-duplicate, keep most recent
    by_id_run: Dict[str, str] = {}
    n_scanned = 0; n_kept = 0; n_skip_op = 0; n_skip_depth = 0; n_skip_fields = 0; n_skip_diff = 0

    run_dirs = sorted(glob.glob(args.run_dirs_glob))
    print(f"[materialize] scanning {len(run_dirs)} run directories", file=sys.stderr)
    for rd in run_dirs:
        run_stamp = os.path.basename(rd)
        for gen_file in sorted(glob.glob(f"{rd}/generation_*.json")):
            try:
                problems = json.load(open(gen_file))
            except Exception:
                continue
            for p in problems:
                n_scanned += 1
                if p.get("op_type") not in ("mutation", "crossover"):
                    n_skip_op += 1; continue
                if not has_required_fields(p):
                    n_skip_fields += 1; continue
                if not is_depth1(p, original_ids):
                    n_skip_depth += 1; continue
                if normalize_difficulty(p.get("difficulty")) in excluded:
                    n_skip_diff += 1; continue
                pid = p.get("id")
                if not pid: continue
                # Most-recent wins
                if pid not in by_id or run_stamp > by_id_run.get(pid, ""):
                    by_id[pid] = p
                    by_id_run[pid] = run_stamp
                n_kept += 1

    seeds = [to_seed_schema(p) for p in by_id.values()]
    seeds.sort(key=lambda s: s["ID"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(seeds, f, ensure_ascii=False, indent=2)

    # Stats summary
    by_diff: Dict[str, int] = {}
    by_op: Dict[str, int] = {}
    by_family: Dict[str, int] = {}
    for p in by_id.values():
        d = p.get("difficulty") or "Medium"
        by_diff[d] = by_diff.get(d, 0) + 1
        op = p.get("op_type") or "?"
        by_op[op] = by_op.get(op, 0) + 1
        # Family from FIRST parent (root family)
        parents = p.get("parent_ids") or []
        if parents:
            fam = parents[0].split("-")[0]
            by_family[fam] = by_family.get(fam, 0) + 1

    print(f"[materialize] scanned={n_scanned}", file=sys.stderr)
    print(f"  filter: skipped op_type={n_skip_op}, fields={n_skip_fields}, depth>1={n_skip_depth}, excluded_diff={n_skip_diff}", file=sys.stderr)
    print(f"  kept (deduplicated): {len(seeds)} seeds → {args.out}", file=sys.stderr)
    print(f"  by difficulty: {by_diff}", file=sys.stderr)
    print(f"  by op_type:    {by_op}", file=sys.stderr)
    print(f"  by family:     {dict(sorted(by_family.items(), key=lambda x: -x[1]))}", file=sys.stderr)


if __name__ == "__main__":
    main()
