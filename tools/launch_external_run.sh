#!/usr/bin/env bash
# Launch one external-benchmark generation run.
# Usage: tools/launch_external_run.sh <bench> <combo> [max_generations]
# Example: tools/launch_external_run.sh math500 combo_a 10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ENTROPYMATH_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DEFAULT_PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
if [[ -x "$DEFAULT_PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
cd "$REPO_DIR"

BENCH="${1:?bench required: math500|aime2025|olympiadbench_en|gsm8k}"
COMBO="${2:?combo required, e.g. combo_a}"
MAX_GEN="${3:-10}"

SEED_FILE="data/seed/external/${BENCH}_seeds.json"
[ "$BENCH" = "olympiadbench_en" ] && SEED_FILE="data/seed/external/olympiadbench_seeds.json"

# Extract seed_ids from the manifest YAML using the local Python environment.
SEED_IDS=$("$PYTHON_BIN" -c "
import yaml
mf = yaml.safe_load(open('data/seed/external/combo_manifest.yaml'))
print(','.join(mf['$BENCH']['$COMBO']['seed_ids']))
")

echo "[launch] bench=$BENCH combo=$COMBO seed_file=$SEED_FILE max_gen=$MAX_GEN"
echo "[launch] seed_ids=$SEED_IDS"

LANGSMITH_PROJECT="entropymath-external-${BENCH}-${COMBO}" \
"$PYTHON_BIN" main.py \
  --seed-file "$SEED_FILE" \
  --seed-ids "$SEED_IDS" \
  --max-generations "$MAX_GEN"
