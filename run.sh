#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-verify}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$CMD" != "verify" ]]; then
  cat >&2 <<'USAGE'
Usage: ./run.sh verify

Code-only smoke test for this repository. Runs offline with the Python
standard library only: no dependency install, no API keys, no model calls.

This repository ships the generation runtime. The released dataset and the
frozen evidence files are hosted separately (see README.md); their integrity
checker is distributed with the supplementary archive, not here.
USAGE
  exit 2
fi

python3 - "$ROOT" <<'PY'
import ast
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
checks = []


def ok(label, detail=""):
    checks.append((True, label, detail))


def fail(label, detail=""):
    checks.append((False, label, detail))


# 1. Runtime entry points and root modules.
root_files = [
    "README.md",
    "requirements.txt",
    "main.py",
    "config.py",
    "tools.py",
    "data_paths.py",
    "artifact_views.py",
    "data/seed/problems.schema.json",
]
missing = [p for p in root_files if not (root / p).exists()]
if missing:
    fail("runtime entry points", f"missing: {missing}")
else:
    ok("runtime entry points", f"{len(root_files)} files")

# 2. Package layout. Every deepagent subpackage must be importable as a
#    regular package, otherwise `from deepagent.nodes... import` breaks.
packages = [
    "deepagent/graph",
    "deepagent/nodes",
    "deepagent/nodes/planning",
    "deepagent/nodes/synthesis",
    "deepagent/nodes/validation",
    "deepagent/nodes/regen",
    "deepagent/nodes/review",
    "prompts",
]
missing_init = [p for p in packages if not (root / p / "__init__.py").exists()]
if missing_init:
    fail("package layout", f"missing __init__.py: {missing_init}")
else:
    ok("package layout", f"{len(packages)} packages")

# 3. Every shipped module parses. Catches truncated or corrupted files
#    without importing third-party dependencies.
sources = sorted(
    p for p in root.rglob("*.py")
    if ".git" not in p.parts and "__pycache__" not in p.parts
)
broken = []
for path in sources:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        broken.append(f"{path.relative_to(root)}:{exc.lineno}: {exc.msg}")
if broken:
    fail("python sources parse", "; ".join(broken[:5]))
else:
    ok("python sources parse", f"{len(sources)} modules")

# 4. System prompts. Each agent role fragment must exist and be non-empty.
prompt_dir = root / "prompts/system"
required_prompts = [
    "mutation_generator.md",
    "crossover_generator.md",
    "validator_execution.md",
    "validator_grounding.md",
    "validator_solvability.md",
    "validator_regen.md",
    "validator_compare.md",
    "validator_anchored_retry.md",
    "grounding_reviewer.md",
    "selector.md",
    "quality.md",
    "synthesis_plan_grouped.md",
    "regen_plan.md",
]
empty_or_missing = [
    name for name in required_prompts
    if not (prompt_dir / name).exists()
    or not (prompt_dir / name).read_text(encoding="utf-8").strip()
]
if empty_or_missing:
    fail("system prompts", f"missing/empty: {empty_or_missing}")
else:
    shipped = sorted(p.name for p in prompt_dir.glob("*.md"))
    ok("system prompts", f"{len(required_prompts)} required, {len(shipped)} shipped")

# 5. Public seed schema. The real seed problems are intentionally not
#    shipped; the schema defines the contract a user's own seed file must meet.
schema_path = root / "data/seed/problems.schema.json"
try:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    item_props = schema.get("items", {}).get("properties", {})
    if schema.get("type") != "array" or not item_props:
        fail("seed schema", "expected an array schema with item properties")
    else:
        ok("seed schema", f"{len(item_props)} item fields")
except (OSError, json.JSONDecodeError) as exc:
    fail("seed schema", str(exc))

# 6. Declared dependencies.
try:
    req_lines = [
        line.strip()
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    core = {"langgraph", "pydantic", "sympy", "numpy"}
    declared = {line.split(">=")[0].split("==")[0].strip().lower() for line in req_lines}
    if not core <= declared:
        fail("requirements.txt", f"missing core packages: {sorted(core - declared)}")
    else:
        ok("requirements.txt", f"{len(req_lines)} pinned requirements")
except OSError as exc:
    fail("requirements.txt", str(exc))

# 7. Anonymity and credential check. Patterns come from the environment so
#    this public source never embeds the identifying strings themselves.
#    Authors set ENTROPY_FORBIDDEN_PATTERNS (colon-separated) before running;
#    reviewers can ignore it, an empty list simply skips the extra patterns.
forbidden = [p for p in os.environ.get("ENTROPY_FORBIDDEN_PATTERNS", "").split(":") if p] + [
    "OPENROUTER_API_" + "KEY=sk-",
    "LANGSMITH_API_" + "KEY=lsv2_",
    "OPENAI_API_" + "KEY=sk-",
]
hits = []
for path in root.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    if path.suffix.lower() in {".png", ".pdf", ".zip", ".pyc"}:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    for needle in forbidden:
        if needle in text:
            hits.append(f"{path.relative_to(root)}: {needle[:24]}...")
if hits:
    fail("anonymity/credential scan", "; ".join(hits[:5]))
else:
    ok("anonymity/credential scan", f"{len(forbidden)} patterns clean")

# Report.
width = max(len(label) for _, label, _ in checks)
for passed, label, detail in checks:
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {label.ljust(width)}  {detail}")

failed = [label for passed, label, _ in checks if not passed]
print()
if failed:
    raise SystemExit(f"code verification FAILED: {failed}")
print(f"code verification passed ({len(checks)}/{len(checks)} checks)")
print()
print("Scope note: this checks the generation runtime shipped in this")
print("repository. The 934-row release is hosted on Hugging Face Datasets and")
print("the frozen evidence files ship with the OpenReview supplementary")
print("archive, which carries its own integrity checker. See README.md.")
PY
