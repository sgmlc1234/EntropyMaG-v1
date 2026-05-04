#!/usr/bin/env python3
"""Build the anonymous OpenReview supplementary artifact staging tree."""

from __future__ import annotations

import csv
import json
import shutil
import stat
from pathlib import Path


def _submission_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "entropymath_neurips2026.tex").exists():
            return parent
    return Path(__file__).resolve().parents[2]


ROOT = _submission_root()
WORKSPACE = ROOT.parent
REPO = WORKSPACE / "repo"
SUPP = ROOT / "supplementary"
ZIP_BASE = ROOT / "supplementary"
ANALYSIS = ROOT / "history" / "analysis"
SCRIPT_SOURCE = ROOT / "history" / "scripts"


IGNORE_NAMES = {
    ".DS_Store",
    ".git",
    ".claude",
    ".env",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "data/runs",
    "data/memory",
    "data/gen_problem",
    "data/seed/private",
    "test",
    "docs",
}


def ignore(dir_path: str, names: list[str]) -> set[str]:
    rel = Path(dir_path).resolve().relative_to(REPO.resolve()) if Path(dir_path).resolve().is_relative_to(REPO.resolve()) else Path()
    ignored = set()
    for name in names:
        child_rel = str((rel / name).as_posix())
        if name in IGNORE_NAMES or child_rel in IGNORE_NAMES:
            ignored.add(name)
    return ignored


def clean() -> None:
    if SUPP.exists():
        shutil.rmtree(SUPP)
    zip_path = ROOT / "supplementary.zip"
    if zip_path.exists():
        zip_path.unlink()
    SUPP.mkdir(parents=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, ignore=ignore)


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def copy_code() -> None:
    code = SUPP / "code"
    for name in [
        "README.md",
        "requirements.txt",
        "main.py",
        "config.py",
        "data_paths.py",
        "artifact_views.py",
        "tools.py",
        ".gitignore",
    ]:
        copy_file(REPO / name, code / name)
    for name in ["deepagent", "prompts", "tools"]:
        copy_tree(REPO / name, code / name)

    schema = REPO / "data/seed/problems.schema.json"
    if schema.exists():
        copy_file(schema, code / "data/seed/problems.schema.json")
    copy_tree(REPO / "data/seed/external", code / "data/seed/external")
    copy_tree(REPO / "data/seed/external_ablation", code / "data/seed/external_ablation")

    paper_scripts = code / "paper_scripts"
    for name in [
        "create_evidence_updates.py",
        "fix_model_eval_summary_labels.py",
        "build_supplementary.py",
    ]:
        copy_file(SCRIPT_SOURCE / name, paper_scripts / name)


def copy_data_sample() -> None:
    data = SUPP / "data_sample"
    copy_tree(REPO / "data/public/entropymath_generated_v1_hf", data / "dataset")
    copy_tree(REPO / "data/public/quality_gate", data / "quality_gate")
    copy_tree(REPO / "data/public/evaluation_samples", data / "evaluation_samples")
    copy_tree(REPO / "data/seed/external", data / "external_eval/seed")
    copy_tree(REPO / "data/seed/external_ablation", data / "external_eval/seed_ablation")
    copy_tree(REPO / "data/eval/external", data / "external_eval/eval")
    external_ablation_eval = REPO / "data/eval/external_ablation"
    if external_ablation_eval.exists():
        copy_tree(external_ablation_eval, data / "external_eval/eval_ablation")
        gen_matrix = external_ablation_eval / "generation_slices/generation_eval_job_matrix.csv"
        if gen_matrix.exists():
            copy_file(gen_matrix, data / "external_eval/eval_ablation/generation_eval_job_matrix.csv")
    copy_tree(ANALYSIS / "model_eval_results/direct_no_tool", data / "model_eval_results/direct_no_tool")
    copy_tree(ANALYSIS / "external_benchmarks", data / "external_benchmarks")
    for rel in [
        "external_significance/external_significance_cells.csv",
        "external_significance/external_significance_comparisons.csv",
        "external_significance/external_significance_summary.md",
        "external_significance/external_expanded_treatment_provenance.csv",
        "external_significance/external_expanded_quality_gate_yield.csv",
        "external_significance/external_treatment_model_summary.csv",
        "external_generation_eval/external_generation_cells.csv",
        "external_generation_eval/external_generation_trends.csv",
        "external_generation_eval/external_generation_summary.md",
    ]:
        src = REPO / "data/analysis" / rel
        if src.exists():
            copy_file(src, data / "external_benchmarks" / src.name)
    for src, name in [
        (REPO / "data/eval/external_ablation/expanded_full_summary.json", "expanded_full_summary.json"),
        (REPO / "data/eval/external_ablation/significance_job_matrix_summary.json", "significance_job_matrix_summary.json"),
        (REPO / "data/eval/external_ablation/generation_slices/generation_eval_job_matrix_summary.json", "generation_eval_job_matrix_summary.json"),
    ]:
        if src.exists():
            copy_file(src, data / "external_benchmarks" / name)
    copy_tree(ANALYSIS / "audit_precheck", data / "audit_precheck")
    ablation_analysis = ANALYSIS / "ablation_microstudy"
    if ablation_analysis.exists():
        copy_tree(ablation_analysis, data / "ablation_microstudy")


def supplementary_readme() -> str:
    return """# EntropyMath Supplementary Artifact

This anonymous OpenReview supplementary artifact contains executable public code and reviewer-facing data samples for the EntropyMath submission.

## Layout

- `code/`: public runtime, prompts, tools, requirements, and paper helper scripts. Local virtual environments, private seeds, saved full run archives, memory caches, logs, and author-local paths are excluded.
- `data_sample/dataset/`: statement-unique EntropyMath-Generated-v1 CSV/JSONL package with README, metadata, Croissant, and license.
- `data_sample/quality_gate/`: clean source CSV, quarantine CSV/JSONL, and quality summary used before release packaging.
- `data_sample/evaluation_samples/`: frozen pre-filter 120-row model-evaluation diagnostic sample and quality-gated 180-row audit sample.
- `data_sample/audit_precheck/`: 30-row Codex-assisted computational/text precheck, pending human confirmation.
- `data_sample/external_eval/`: external benchmark seed controls, combo manifest, 20-control arms, 250-row treatment arms, and evaluation job matrices.
- `data_sample/external_eval/seed_ablation/`: 20-seed-per-benchmark ablation/new-seed expansion manifests for MATH-500, AIME 2025, and GSM8K.
- `data_sample/external_eval/eval_ablation/`: expanded external-significance control/treatment JSONL files and job matrices for 20-control/250-treatment significance and generation-slice evaluation.
- `data_sample/model_eval_results/`: model-evaluation raw outputs, summaries, flattened CSVs, and run manifests.
- `data_sample/external_benchmarks/`: aggregate external benchmark summaries, expanded significance and generation-slice analysis tables, treatment provenance and quality-gate yield summaries, control uncertainty checks, earlier quality-flag manifest, trace-list exports, and operational metadata summaries.

## Verification

Run the reduced verification command:

```bash
./run.sh verify
```

This checks package structure, metadata-derived dataset row counts, quality-gate invariants, audit-precheck counts, expanded 20-control/250-treatment external files, significance and generation job-matrix readiness, generation/significance analysis rows, expanded treatment quality-yield summaries, ablation seed manifests, earlier external quality flags, and model-summary label traceability. It does not rerun long generation jobs or call model APIs.
"""


def run_sh() -> str:
    return r"""#!/usr/bin/env bash
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
    "data_sample/external_benchmarks/external_generation_cells.csv",
    "data_sample/external_benchmarks/external_generation_trends.csv",
    "data_sample/external_benchmarks/external_expanded_treatment_provenance.csv",
    "data_sample/external_benchmarks/external_expanded_quality_gate_yield.csv",
    "data_sample/external_benchmarks/external_treatment_model_summary.csv",
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

with (root / "data_sample/external_benchmarks/external_generation_cells.csv").open(encoding="utf-8", newline="") as handle:
    gen_cells = list(csv.DictReader(handle))
assert len(gen_cells) == 90

with (root / "data_sample/external_benchmarks/external_treatment_model_summary.csv").open(encoding="utf-8", newline="") as handle:
    treatment_model_rows = list(csv.DictReader(handle))
assert len(treatment_model_rows) == 3
assert {int(row["problems"]) for row in treatment_model_rows} == {750}

with (root / "data_sample/external_benchmarks/external_expanded_quality_gate_yield.csv").open(encoding="utf-8", newline="") as handle:
    yield_rows = list(csv.DictReader(handle))
yield_total = next(row for row in yield_rows if row["benchmark"] == "total")
assert int(yield_total["source_before_gate"]) == 864
assert int(yield_total["quality_kept"]) == 800
assert int(yield_total["usable_pool"]) == 795
assert int(yield_total["treatment_cap"]) == 750

sig_jobs = json.loads((root / "data_sample/external_benchmarks/significance_job_matrix_summary.json").read_text(encoding="utf-8"))
assert sig_jobs["jobs"] == 18 and sig_jobs["ready_jobs"] == 18
gen_jobs = json.loads((root / "data_sample/external_benchmarks/generation_eval_job_matrix_summary.json").read_text(encoding="utf-8"))
assert gen_jobs["jobs"] == 90 and gen_jobs["ready_jobs"] == 90

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
    assert count_jsonl(f"data_sample/external_eval/eval_ablation/{bench}_expanded_full_treatment_250.jsonl") == 250

# Author-side anti-leak check. Patterns are loaded from environment variables
# so this public source never embeds the actual identifying strings.
# Authors set ENTROPY_FORBIDDEN_PATTERNS as a colon-separated list before
# running the build locally; reviewers can ignore (no patterns => skip).
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
"""


def validate_inputs() -> None:
    with (REPO / "data/public/entropymath_generated_v1_hf/metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (REPO / "data/public/quality_gate/quality_summary.json").open(encoding="utf-8") as handle:
        quality = json.load(handle)
    if int(metadata.get("row_count", 0)) <= 0:
        raise ValueError("Dataset row_count must be positive")
    if int(metadata.get("row_count", 0)) > int(quality.get("kept_row_count", 0)):
        raise ValueError("Dataset row_count exceeds quality-gate kept row count")
    with (ANALYSIS / "audit_precheck/codex_precheck_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("n") != 30:
        raise ValueError("Unexpected audit precheck size")
    if not (ANALYSIS / "external_benchmarks/external_quality_flags_summary.json").exists():
        raise ValueError("Missing external quality-flag summary")
    with (ANALYSIS / "model_eval_results/direct_no_tool/combined_summary_flat.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        models = {row["model"] for row in csv.DictReader(handle)}
    if "." in models:
        raise ValueError("combined_summary_flat.csv still contains model='.'")


def main() -> None:
    validate_inputs()
    clean()
    copy_code()
    copy_data_sample()
    write_text(SUPP / "README.md", supplementary_readme())
    write_text(SUPP / "run.sh", run_sh(), executable=True)
    archive = shutil.make_archive(str(ZIP_BASE), "zip", root_dir=SUPP)
    print(json.dumps({"supplementary": str(SUPP), "zip": archive}, indent=2))


if __name__ == "__main__":
    main()
