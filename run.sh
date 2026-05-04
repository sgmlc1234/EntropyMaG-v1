#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-verify}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$CMD" != "verify" ]]; then
  echo "Usage: ./run.sh verify" >&2
  exit 2
fi

python3 - "$ROOT" <<'PY'
import csv
import json
import os
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

required = [
    "README.md",
    "code/README.md",
    "code/requirements.txt",
    "code/main.py",
    "code/tools/analyze_ablation_microstudy.py",
    "code/tools/build_ablation_seed_files.py",
    "code/tools/launch_ablation_run.sh",
    "code/tools/launch_new_seed_treatment_campaign.sh",
    "code/tools/apply_quality_gate.py",
    "code/tools/package_hf_release.py",
    "code/tools/analyze_external_significance.py",
    "code/tools/analyze_external_generation_eval.py",
    "code/tools/make_external_eval_paper_figures.py",
    "code/data/seed/external_ablation/combo_manifest_ablation.yaml",
    "data_sample/dataset/entropymath_generated_v1.csv",
    "data_sample/dataset/entropymath_generated_v1.jsonl",
    "data_sample/dataset/croissant.json",
    "data_sample/dataset/metadata.json",
    "data_sample/quality_gate/quality_summary.json",
    "data_sample/quality_gate/entropymath_quality_clean.csv",
    "data_sample/quality_gate/entropymath_quality_quarantine.csv",
    "data_sample/quality_gate/entropymath_quality_quarantine.jsonl",
    "data_sample/evaluation_samples/model_eval_stratified_120.csv",
    "data_sample/evaluation_samples/human_audit_stratified_180.csv",
    "data_sample/audit_precheck/codex_precheck_30.csv",
    "data_sample/audit_precheck/codex_precheck_summary.json",
    "data_sample/external_eval/seed_ablation/combo_manifest_ablation.yaml",
    "data_sample/external_eval/seed_ablation/math500_ablation20_seeds.json",
    "data_sample/external_eval/seed_ablation/aime2025_ablation20_seeds.json",
    "data_sample/external_eval/seed_ablation/gsm8k_ablation20_seeds.json",
    "data_sample/external_eval/eval_ablation/math500_control_20.jsonl",
    "data_sample/external_eval/eval_ablation/aime2025_control_20.jsonl",
    "data_sample/external_eval/eval_ablation/gsm8k_control_20.jsonl",
    "data_sample/external_eval/eval_ablation/math500_expanded_full_treatment_250.jsonl",
    "data_sample/external_eval/eval_ablation/aime2025_expanded_full_treatment_250.jsonl",
    "data_sample/external_eval/eval_ablation/gsm8k_expanded_full_treatment_250.jsonl",
    "data_sample/external_eval/eval_ablation/math500_new_control_10.jsonl",
    "data_sample/external_eval/eval_ablation/aime2025_new_control_10.jsonl",
    "data_sample/external_eval/eval_ablation/gsm8k_new_control_10.jsonl",
    "data_sample/external_eval/eval_ablation/significance_job_matrix.csv",
    "data_sample/external_eval/eval_ablation/generation_eval_job_matrix.csv",
    "data_sample/external_benchmarks/external_control_ci_by_benchmark.csv",
    "data_sample/external_benchmarks/external_three_run_reanalysis.csv",
    "data_sample/external_benchmarks/external_quality_flags.csv",
    "data_sample/external_benchmarks/external_quality_flags_summary.json",
    "data_sample/external_benchmarks/external_significance_cells.csv",
    "data_sample/external_benchmarks/external_significance_comparisons.csv",
    "data_sample/external_benchmarks/external_significance_summary.md",
    "data_sample/external_benchmarks/external_generation_cells.csv",
    "data_sample/external_benchmarks/external_generation_trends.csv",
    "data_sample/external_benchmarks/external_generation_summary.md",
    "data_sample/external_benchmarks/external_expanded_treatment_provenance.csv",
    "data_sample/external_benchmarks/external_expanded_quality_gate_yield.csv",
    "data_sample/external_benchmarks/external_treatment_model_summary.csv",
    "data_sample/external_benchmarks/expanded_full_summary.json",
    "data_sample/external_benchmarks/significance_job_matrix_summary.json",
    "data_sample/external_benchmarks/generation_eval_job_matrix_summary.json",
    "data_sample/model_eval_results/direct_no_tool/combined_summary_flat.csv",
]
missing = [p for p in required if not (root / p).exists()]
if missing:
    raise SystemExit(f"missing required files: {missing}")

def count_csv(path):
    with (root / path).open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))

def count_jsonl(path):
    with (root / path).open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())

def read_jsonl(path):
    with (root / path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

metadata = json.loads((root / "data_sample/dataset/metadata.json").read_text(encoding="utf-8"))
quality = json.loads((root / "data_sample/quality_gate/quality_summary.json").read_text(encoding="utf-8"))
row_count = int(metadata["row_count"])
kept_count = int(quality["kept_row_count"])
excluded_count = int(quality["excluded_row_count"])
assert row_count > 0
assert kept_count >= row_count
assert excluded_count == int(quality["hard_exclude_count"]) + int(quality["support_gap_exclude_count"])
assert count_csv("data_sample/dataset/entropymath_generated_v1.csv") == row_count
assert count_jsonl("data_sample/dataset/entropymath_generated_v1.jsonl") == row_count
assert count_csv("data_sample/evaluation_samples/model_eval_stratified_120.csv") == 120
assert count_csv("data_sample/evaluation_samples/human_audit_stratified_180.csv") == 180
assert count_csv("data_sample/audit_precheck/codex_precheck_30.csv") == 30

summary = json.loads((root / "data_sample/audit_precheck/codex_precheck_summary.json").read_text(encoding="utf-8"))
assert summary["n"] == 30

quality_patterns = [
    ("provided_answer_incorrect", r"\bprovided\s+(?:final\s+)?answer\b.{0,120}\bincorrect\b"),
    ("candidate_answer_incorrect", r"\bcandidate(?:\s+answer)?\b.{0,120}\bincorrect\b"),
    ("final_answer_incorrect", r"\bfinal\s+answer\b.{0,120}\bincorrect\b"),
    ("sandbox_evidence", r"\bsandbox\s+evidence\b"),
    ("sandbox_output", r"\bsandbox\s+output\b"),
    ("tool_evidence", r"\btool\s+evidence\b"),
    ("wait_recheck", r"\bwait[,;:]?\s+"),
    ("re_evaluating", r"\bre[- ]?evaluat(?:e|ing|ion)\b"),
]
with (root / "data_sample/dataset/entropymath_generated_v1.csv").open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        text = row.get("solution", "")
        for name, pattern in quality_patterns:
            if re.search(pattern, text, re.I | re.S):
                raise AssertionError(f"quality pattern survived in clean release: {name} {row.get('release_id')}")

with (root / "data_sample/model_eval_results/direct_no_tool/combined_summary_flat.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
models = {row["model"] for row in rows}
expected = {
    "openai/gpt-5.4-mini",
    "google/gemini-3.1-flash-lite-preview",
    "anthropic/claude-haiku-4.5",
}
assert "." not in models, models
assert expected <= models, models

with (root / "data_sample/external_benchmarks/external_control_ci_by_benchmark.csv").open(encoding="utf-8", newline="") as handle:
    ci_rows = list(csv.DictReader(handle))
assert len(ci_rows) == 3

external_flags = json.loads(
    (root / "data_sample/external_benchmarks/external_quality_flags_summary.json").read_text(encoding="utf-8")
)
assert external_flags["total_row_count"] > 0
assert external_flags["flagged_row_count"] >= 0

with (root / "data_sample/external_benchmarks/external_significance_comparisons.csv").open(encoding="utf-8", newline="") as handle:
    sig_rows = list(csv.DictReader(handle))
assert len(sig_rows) == 9
assert {int(row["control_problems"]) for row in sig_rows} == {20}
assert {int(row["treatment_problems"]) for row in sig_rows} == {250}
sig_by_bench = {}
for row in sig_rows:
    sig_by_bench.setdefault(row["benchmark"], []).append(row)
assert set(sig_by_bench) == {"math500", "aime2025", "gsm8k"}
assert all(float(row["problem_accuracy_drop_pp"]) > 0 for row in sig_by_bench["math500"])
assert all(float(row["problem_accuracy_drop_pp"]) > 0 for row in sig_by_bench["gsm8k"])
assert sum(float(row["problem_accuracy_boot95_low_pp"]) > 0 for row in sig_by_bench["gsm8k"]) == 3

with (root / "data_sample/external_benchmarks/external_significance_cells.csv").open(encoding="utf-8", newline="") as handle:
    sig_cells = list(csv.DictReader(handle))
assert len(sig_cells) == 18

with (root / "data_sample/external_benchmarks/external_generation_cells.csv").open(encoding="utf-8", newline="") as handle:
    gen_cells = list(csv.DictReader(handle))
assert len(gen_cells) == 90
assert {int(row["generation"]) for row in gen_cells} == set(range(1, 11))

with (root / "data_sample/external_benchmarks/external_generation_trends.csv").open(encoding="utf-8", newline="") as handle:
    gen_trends = list(csv.DictReader(handle))
assert len(gen_trends) == 9

with (root / "data_sample/external_benchmarks/external_treatment_model_summary.csv").open(encoding="utf-8", newline="") as handle:
    treatment_model_rows = list(csv.DictReader(handle))
assert len(treatment_model_rows) == 3
assert {int(row["problems"]) for row in treatment_model_rows} == {750}
assert {int(row["runs"]) for row in treatment_model_rows} == {2250}

with (root / "data_sample/external_benchmarks/external_expanded_quality_gate_yield.csv").open(encoding="utf-8", newline="") as handle:
    yield_rows = list(csv.DictReader(handle))
yield_total = next(row for row in yield_rows if row["benchmark"] == "total")
assert int(yield_total["source_before_gate"]) == 864
assert int(yield_total["quality_kept"]) == 800
assert int(yield_total["usable_pool"]) == 795
assert int(yield_total["treatment_cap"]) == 750

sig_jobs = json.loads((root / "data_sample/external_benchmarks/significance_job_matrix_summary.json").read_text(encoding="utf-8"))
assert sig_jobs["jobs"] == 18
assert sig_jobs["ready_jobs"] == 18
assert sig_jobs["not_ready_jobs"] == 0

gen_jobs = json.loads((root / "data_sample/external_benchmarks/generation_eval_job_matrix_summary.json").read_text(encoding="utf-8"))
assert gen_jobs["jobs"] == 90
assert gen_jobs["ready_jobs"] == 90
assert gen_jobs["not_ready_jobs"] == 0

for bench in ["math500", "aime2025", "gsm8k"]:
    seed_path = root / f"data_sample/external_eval/seed_ablation/{bench}_ablation20_seeds.json"
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    assert len(seeds) == 20
    splits = {}
    for seed in seeds:
        splits[seed.get("_ablation_seed_split", "")] = splits.get(seed.get("_ablation_seed_split", ""), 0) + 1
    assert splits == {"existing_matched": 10, "new_diagnostic": 10}, (bench, splits)
    assert count_jsonl(f"data_sample/external_eval/eval_ablation/{bench}_control_20.jsonl") == 20
    assert count_jsonl(f"data_sample/external_eval/eval_ablation/{bench}_new_control_10.jsonl") == 10
    treatment_rows = read_jsonl(f"data_sample/external_eval/eval_ablation/{bench}_expanded_full_treatment_250.jsonl")
    assert len(treatment_rows) == 250
    assert len({row["_statement_sha256"] for row in treatment_rows}) == 250
    assert all(row.get("_arm") == "treatment" for row in treatment_rows)

with (root / "data_sample/external_benchmarks/external_expanded_treatment_provenance.csv").open(encoding="utf-8", newline="") as handle:
    provenance_rows = list(csv.DictReader(handle))
total = next(row for row in provenance_rows if row["benchmark"] == "total")
assert int(total["treatment_rows"]) == 750
assert int(total["published_existing"]) == 334
assert int(total["generated_full"]) == 400
assert int(total["partial_recovered"]) == 16

# Author-side anti-leak check. Patterns are loaded from environment variables
# so this public source never embeds the actual identifying strings.
# Authors set ENTROPY_FORBIDDEN_PATTERNS as a colon-separated list before
# running `run.sh verify` locally; reviewers can ignore (no patterns => skip).
forbidden = [p for p in os.environ.get("ENTROPY_FORBIDDEN_PATTERNS", "").split(":") if p] + [
    "OPENROUTER_API_" + "KEY=sk-",
    "LANGSMITH_API_" + "KEY=lsv2_",
]
hits = []
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix.lower() in {".png", ".pdf", ".zip"}:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    for needle in forbidden:
        if needle in text:
            hits.append(f"{path.relative_to(root)}: {needle}")
if hits:
    raise SystemExit("forbidden strings found:\n" + "\n".join(hits[:20]))

print("supplementary verification passed")
PY
