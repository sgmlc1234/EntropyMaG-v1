#!/usr/bin/env python3
"""Create a local Hugging Face-style release bundle for EntropyMath."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


REQUIRED_COLUMNS = ["id", "statement", "answer", "solution", "verification_code"]
DEFAULT_DATASET_URL = "https://openreview.net/attachment?id=ANONYMOUS_SUBMISSION&name=supplementary.zip"


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [column for column in REQUIRED_COLUMNS if column not in (rows[0].keys() if rows else [])]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    bad = [row.get("id", f"row_{idx}") for idx, row in enumerate(rows, 1) if any(not row.get(c) for c in REQUIRED_COLUMNS)]
    if bad:
        raise ValueError(f"{len(bad)} rows are missing required values; first bad ids: {bad[:5]}")
    return rows


def _statement_unique(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Collapse release rows to one row per generated statement."""
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("source_run", ""),
            row.get("generation", ""),
            row.get("id", ""),
            row.get("statement_sha256", ""),
        ),
    )
    seen = set()
    unique: List[Dict[str, str]] = []
    for row in ordered:
        key = row.get("statement_sha256") or hashlib.sha256(row.get("statement", "").encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _release_id(row: Dict[str, str]) -> str:
    statement_hash = row.get("statement_sha256") or hashlib.sha256(row.get("statement", "").encode("utf-8")).hexdigest()
    return f"emv1_{statement_hash[:16]}"


def _with_release_ids(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        new_row = dict(row)
        new_row["release_id"] = _release_id(row)
        out.append(new_row)
    return out


def _columns(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    columns = list(rows[0].keys())
    if "release_id" in columns:
        columns = ["release_id"] + [column for column in columns if column != "release_id"]
    return columns


def _write_csv(path: Path, rows: List[Dict[str, str]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_quality_summary(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _croissant(
    dataset_name: str,
    rows: List[Dict[str, str]],
    csv_name: str,
    dataset_url: str,
    quality_summary: Dict,
) -> Dict:
    columns = _columns(rows)
    return {
        "@context": {
            "@language": "en",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "prov": "http://www.w3.org/ns/prov#",
            "dct": "http://purl.org/dc/terms/",
        },
        "@type": "sc:Dataset",
        "name": dataset_name,
        "description": (
            "EntropyMath-Generated-v1 is a quality-gated, statement-unique generated mathematical "
            "reasoning dataset with statements, answers, solutions, verification_code computational "
            "consistency evidence, and provenance fields."
        ),
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": dataset_url,
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "datePublished": datetime.now(timezone.utc).date().isoformat(),
        "version": "1.0.0",
        "keywords": ["mathematical reasoning", "evaluation", "benchmark generation", "LLM evaluation"],
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "dataset_csv",
                "name": csv_name,
                "contentUrl": csv_name,
                "encodingFormat": "text/csv",
                "sha256": "",
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "name": "problems",
                "data": {"@id": "dataset_csv"},
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": column,
                        "name": column,
                        "dataType": "sc:Text" if column != "rai:hasSyntheticData" else "sc:Boolean",
                        "source": {"fileObject": {"@id": "dataset_csv"}, "extract": {"column": column}},
                    }
                    for column in columns
                ],
            }
        ],
        "rai:dataCollection": (
            "Generated from saved EntropyMath deep-run artifacts by export, completeness filtering, "
            "conservative quality gating for contradiction/support-gap rows, statement hashing, "
            "statement-level deduplication, and Hugging Face-style packaging scripts."
        ),
        "rai:dataAnnotationProtocol": (
            "Each released row contains generated solution text and verification_code as computational "
            "consistency evidence. Rows with explicit answer-contradiction cues, verification-code failures "
            "or stdout-answer disagreements, and support-gap cues were excluded by a pre-release quality "
            "gate. The release provides frozen stratified samples for model-evaluation diagnostics and "
            "audit precheck; the current public package does not report completed human-audit outcomes."
        ),
        "rai:personalSensitiveInformation": "No personal data is intentionally included.",
        "rai:hasSyntheticData": True,
        "rai:dataBiases": (
            "The dataset is biased toward competition-style mathematics and families represented "
            "in the private seed pool and generator prompts. Difficulty labels are generator-side labels "
            "and should not be interpreted as externally calibrated difficulty estimates."
        ),
        "rai:dataLimitations": (
            "verification_code checks computational consistency but is not a proof-level guarantee. "
            "LLM-assisted validation can miss ambiguity, shortcut solutions, or mathematically invalid derivations. "
            "The package includes answers and verification_code, so it is not suitable as a permanent hidden leaderboard "
            "after release or for high-stakes assessment without independent audit. The quarantine manifest records "
            "rows excluded by the quality gate and should be treated as diagnostic provenance, not released benchmark data."
        ),
        "rai:dataUseCases": (
            "Intended uses are mathematical reasoning evaluation research, generator audits, benchmark-methodology "
            "studies, reproducibility checks, and stress-testing evaluation workflows. Validity has not been established "
            "for high-stakes educational assessment, proof certification, or claims of contamination-free hidden evaluation."
        ),
        "rai:dataSocialImpact": (
            "Positive impacts include more transparent generated-evaluation artifacts and clearer audit trails. "
            "Negative risks include overfitting to answer-visible generated data, overstating mathematical correctness, "
            "or using synthetic problems as high-stakes assessments. Mitigations include provenance hashes, explicit "
            "limitations, answer-visible release labeling, and frozen audit/evaluation samples."
        ),
        "prov:wasDerivedFrom": ["private EntropyMath seed pool and saved deep-run artifacts"],
        "prov:wasGeneratedBy": [
            "tools/export_validated_deep_runs_csv.py",
            "tools/apply_quality_gate.py",
            "tools/package_hf_release.py",
            "tools/make_stratified_samples.py",
        ],
        "rai:qualityGate": {
            "source_row_count": quality_summary.get("source_row_count"),
            "kept_row_count": quality_summary.get("kept_row_count"),
            "excluded_row_count": quality_summary.get("excluded_row_count"),
            "hard_exclude_count": quality_summary.get("hard_exclude_count"),
            "support_gap_exclude_count": quality_summary.get("support_gap_exclude_count"),
            "quarantine_manifest": quality_summary.get("quarantine_jsonl"),
        },
    }


def build(args: argparse.Namespace) -> Dict:
    source = args.source_csv.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_rows = _read_rows(source)
    quality_summary = _load_quality_summary(args.quality_summary.expanduser().resolve())
    rows = _with_release_ids(_statement_unique(source_rows))
    columns = _columns(rows)

    csv_path = out / "entropymath_generated_v1.csv"
    jsonl_path = out / "entropymath_generated_v1.jsonl"
    _write_csv(csv_path, rows, columns)
    _write_jsonl(jsonl_path, rows)

    statement_counts = Counter(row.get("statement_sha256") for row in source_rows)
    duplicate_statement_groups = sum(1 for count in statement_counts.values() if count > 1)
    operation_counts = dict(Counter(row.get("operation") or "unknown" for row in rows))
    difficulty_counts = dict(Counter(row.get("difficulty_label") or "unknown" for row in rows))
    metadata = {
        "dataset_name": "EntropyMath-Generated-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_url": args.dataset_url,
        "dataset_url_is_placeholder": args.dataset_url == DEFAULT_DATASET_URL,
        "source_csv": str(args.source_csv),
        "source_complete_row_count": quality_summary.get("source_row_count", len(source_rows)),
        "source_clean_row_count": len(source_rows),
        "row_count": len(rows),
        "statement_unique_count": len(rows),
        "duplicate_statement_group_count": duplicate_statement_groups,
        "license": "CC BY 4.0",
        "columns": columns,
        "operation_counts": operation_counts,
        "difficulty_counts": difficulty_counts,
        "files": {
            "csv": csv_path.name,
            "jsonl": jsonl_path.name,
            "croissant": "croissant.json",
            "readme": "README.md",
            "license": "LICENSE",
        },
        "quality_gate": {
            "summary_path": str(args.quality_summary),
            "source_row_count": quality_summary.get("source_row_count", len(source_rows)),
            "processed_row_count": quality_summary.get("processed_row_count", len(source_rows)),
            "kept_row_count": quality_summary.get("kept_row_count", len(source_rows)),
            "excluded_row_count": quality_summary.get("excluded_row_count", 0),
            "hard_exclude_count": quality_summary.get("hard_exclude_count", 0),
            "support_gap_exclude_count": quality_summary.get("support_gap_exclude_count", 0),
            "exclusion_reason_counts": quality_summary.get("exclusion_reason_counts", {}),
            "clean_csv": quality_summary.get("clean_csv", str(args.source_csv)),
            "quarantine_csv": quality_summary.get("quarantine_csv"),
            "quarantine_jsonl": quality_summary.get("quarantine_jsonl"),
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    croissant = _croissant("EntropyMath-Generated-v1", rows, csv_path.name, args.dataset_url, quality_summary)
    croissant["distribution"][0]["sha256"] = _sha256_file(csv_path)
    (out / "croissant.json").write_text(json.dumps(croissant, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "LICENSE").write_text(
        "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        "See https://creativecommons.org/licenses/by/4.0/\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "\n".join(
            [
                "---",
                "license: cc-by-4.0",
                "language:",
                "- en",
                "pretty_name: EntropyMath v1",
                "task_categories:",
                "- question-answering",
                "tags:",
                "- mathematical-reasoning",
                "- synthetic-data",
                "- evaluation",
                "- benchmark",
                "size_categories:",
                "- 1K<n<10K",
                "---",
                "",
                "# EntropyMath-Generated-v1",
                "",
                "EntropyMath-Generated-v1 is a quality-gated generated mathematical reasoning evaluation resource.",
                f"It contains {len(rows):,} statement-unique generated problems exported from historical EntropyMath `*-deep-run` artifacts.",
                "Before packaging, a conservative quality gate excludes rows with explicit answer-contradiction cues, verification-code failures or stdout-answer disagreements, and support-gap cues.",
                f"The gate kept {metadata['quality_gate']['kept_row_count']:,} of {metadata['quality_gate']['source_row_count']:,} source rows before statement-level deduplication and quarantined {metadata['quality_gate']['excluded_row_count']:,} rows.",
                "",
                "## Files",
                "",
                "- `entropymath_generated_v1.csv`: canonical release table",
                "- `entropymath_generated_v1.jsonl`: JSONL mirror",
                "- `croissant.json`: MLCommons Croissant metadata with Responsible AI fields",
                "- `metadata.json`: local packaging statistics, including quality-gate counts",
                "- `LICENSE`: CC BY 4.0 notice",
                "- `quality_gate/`: clean source CSV, quarantine manifest, and quality-gate summary in the supplementary artifact",
                "- `evaluation_samples/model_eval_stratified_120.csv`: frozen pre-filter model-evaluation diagnostic sample",
                "- `evaluation_samples/human_audit_stratified_180.csv`: frozen quality-gated audit sample",
                "- `external_eval/seed/`: external benchmark seed controls and combo manifest",
                "- `external_eval/eval/`: external benchmark control/treatment JSONL arms",
                "- `external_benchmarks/results_summary.md`: aggregate external benchmark transformation summary",
                "",
                "## Columns",
                "",
                ", ".join(columns),
                "",
                "`release_id` is the unique row key for this release. The original `id` field is a lineage-readable generator label and is not unique.",
                "",
                "## Reviewer Access",
                "",
                f"Croissant `url`: `{args.dataset_url}`.",
                "For anonymous review, this directory is distributed inside the OpenReview supplementary artifact together with executable code and verification scripts.",
                "",
                "## Intended Use",
                "",
                "The dataset is intended for mathematical reasoning evaluation research, generator audit, and benchmark-methodology analysis.",
                "It is not a proof-certified mathematical corpus and requires independent audit before high-stakes use.",
                "Because answers, solutions, and verification_code are released, this package is not suitable as a permanent hidden leaderboard.",
                "",
                "## Known Limitations",
                "",
                "- Problems are generated and skew toward competition-style mathematics.",
                "- `verification_code` checks computational consistency but does not prove every solution.",
                "- LLM-assisted validation can miss ambiguity, shortcut solutions, or invalid derivations.",
                "- The quarantine manifest is provided for diagnostic transparency and is not part of the clean benchmark release.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("data/public/quality_gate/entropymath_quality_clean.csv"),
    )
    parser.add_argument(
        "--quality-summary",
        type=Path,
        default=Path("data/public/quality_gate/quality_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/public/entropymath_generated_v1_hf"),
    )
    parser.add_argument(
        "--dataset-url",
        default=DEFAULT_DATASET_URL,
        help="Canonical reviewer-accessible dataset URL to place in Croissant metadata.",
    )
    print(json.dumps(build(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
