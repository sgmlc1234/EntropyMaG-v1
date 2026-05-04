#!/usr/bin/env bash
# Generate an additional treatment campaign from the 10 new external seeds.
# Usage: tools/launch_new_seed_treatment_campaign.sh <bench> [target_total] [max_generations] [--dry-run]
# Example: tools/launch_new_seed_treatment_campaign.sh math500 100 10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${1:?bench required: math500|aime2025|gsm8k}"
TARGET_TOTAL="${2:-100}"
MAX_GEN="${3:-10}"
DRY_ARG="${4:-}"
COMBOS=(new_a new_b new_c new_d)

PER_COMBO=$(( (TARGET_TOTAL + ${#COMBOS[@]} - 1) / ${#COMBOS[@]} ))

echo "[new-seed-campaign] bench=$BENCH target_total=$TARGET_TOTAL max_gen=$MAX_GEN per_combo=$PER_COMBO"
for combo in "${COMBOS[@]}"; do
  if [[ "$DRY_ARG" == "--dry-run" ]]; then
    "${SCRIPT_DIR}/launch_ablation_run.sh" "$BENCH" "$combo" full "$MAX_GEN" "$PER_COMBO" --dry-run
  else
    "${SCRIPT_DIR}/launch_ablation_run.sh" "$BENCH" "$combo" full "$MAX_GEN" "$PER_COMBO"
  fi
done
