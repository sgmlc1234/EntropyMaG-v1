#!/usr/bin/env bash
# Launch one external ablation or new-seed expansion run.
# Usage: tools/launch_ablation_run.sh <bench> <combo> <condition> [max_generations] [target_problem_count] [--dry-run]
# Example: tools/launch_ablation_run.sh math500 existing_a no_near_copy 2
# Example: tools/launch_ablation_run.sh math500 new_a full 10 25

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ENTROPYMATH_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DEFAULT_PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
if [[ -x "$DEFAULT_PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
MANIFEST="${ENTROPYMATH_ABLATION_MANIFEST:-data/seed/external_ablation/combo_manifest_ablation.yaml}"
cd "$REPO_DIR"

BENCH="${1:?bench required: math500|aime2025|gsm8k}"
COMBO="${2:?combo required, e.g. existing_a|new_a}"
CONDITION="${3:?condition required: full|no_near_copy|no_solvability}"
MAX_GEN="${4:-2}"
TARGET_COUNT="${5:-0}"
DRY_RUN=0
if [[ "$MAX_GEN" == "--dry-run" ]]; then
  DRY_RUN=1
  MAX_GEN=2
fi
if [[ "$TARGET_COUNT" == "--dry-run" ]]; then
  DRY_RUN=1
  TARGET_COUNT=0
fi
if [[ "${6:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

case "$CONDITION" in
  full|no_near_copy|no_solvability) ;;
  *)
    echo "condition must be one of: full, no_near_copy, no_solvability" >&2
    exit 2
    ;;
esac

IFS=$'\t' read -r SEED_FILE SEED_IDS SEED_SPLIT ROLE < <("$PYTHON_BIN" - "$MANIFEST" "$BENCH" "$COMBO" <<'PY'
import sys
import yaml

manifest_path, bench, combo = sys.argv[1:4]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = yaml.safe_load(handle)
try:
    bench_entry = manifest[bench]
    combo_entry = bench_entry[combo]
except KeyError as exc:
    raise SystemExit(f"unknown manifest key: {exc}") from exc
print(
    "\t".join(
        [
            bench_entry["seed_file"],
            ",".join(combo_entry["seed_ids"]),
            combo_entry.get("split", ""),
            combo_entry.get("target_treatment_role", ""),
        ]
    )
)
PY
)

ARGS=(
  main.py
  --seed-file "$SEED_FILE"
  --seed-ids "$SEED_IDS"
  --max-generations "$MAX_GEN"
  --ablation-condition "$CONDITION"
)

if [[ "$TARGET_COUNT" != "0" && "$TARGET_COUNT" != "" ]]; then
  ARGS+=(--target-problem-count "$TARGET_COUNT")
fi

echo "[launch-ablation] bench=$BENCH combo=$COMBO split=$SEED_SPLIT role=$ROLE condition=$CONDITION max_gen=$MAX_GEN target_count=$TARGET_COUNT"
echo "[launch-ablation] seed_file=$SEED_FILE"
echo "[launch-ablation] seed_ids=$SEED_IDS"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[launch-ablation] dry_run command: LANGSMITH_PROJECT=%q %q' \
    "entropymath-ablation-${BENCH}-${COMBO}-${CONDITION}" "$PYTHON_BIN"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

LANGSMITH_PROJECT="entropymath-ablation-${BENCH}-${COMBO}-${CONDITION}" \
"$PYTHON_BIN" "${ARGS[@]}"
