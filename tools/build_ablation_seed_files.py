#!/usr/bin/env python3
"""Build external-ablation seed files and combo manifest.

The script preserves the existing 10 public external seeds per benchmark and
adds 10 new reviewer-traceable seeds from public benchmark sources. New seeds
are staged separately under data/seed/external_ablation so existing paper
artifacts remain stable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
import yaml


REPO = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO / "data/seed/external"
OUT_DIR = REPO / "data/seed/external_ablation"

HF_ROWS = "https://datasets-server.huggingface.co/rows"
MATH500_ROWS_URL = (
    f"{HF_ROWS}?dataset=HuggingFaceH4/MATH-500&config=default&split=test&offset={{offset}}&length={{length}}"
)
GSM8K_ROWS_URL = (
    f"{HF_ROWS}?dataset=openai/gsm8k&config=main&split=test&offset={{offset}}&length={{length}}"
)
AIME2025_URLS = {
    "I": "https://huggingface.co/datasets/opencompass/AIME2025/resolve/main/aime2025-I.jsonl",
    "II": "https://huggingface.co/datasets/opencompass/AIME2025/resolve/main/aime2025-II.jsonl",
}

SUBJECT_PREFIX = {
    "Algebra": "algebr",
    "Number Theory": "number",
    "Counting & Probability": "counti",
    "Geometry": "geomet",
    "Precalculus": "precal",
}
MATH_SUBJECT_ORDER = ["Algebra", "Number Theory", "Counting & Probability", "Geometry", "Precalculus"]


def _get_json(url: str) -> Dict[str, Any]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _get_text(url: str) -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def _print_answer_code(answer: Any) -> str:
    return f"print({str(answer)!r})\n"


def _load_existing(name: str) -> List[Dict[str, Any]]:
    rows = json.loads((SOURCE_DIR / name).read_text(encoding="utf-8"))
    return [dict(row, _ablation_seed_split="existing_matched") for row in rows]


def _write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fetch_rows(url_template: str, *, total: int, chunk: int = 100) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for offset in range(0, total, chunk):
        payload = _get_json(url_template.format(offset=offset, length=min(chunk, total - offset)))
        for item in payload.get("rows", []):
            row = dict(item.get("row", {}) or {})
            row["_row_idx"] = item.get("row_idx")
            rows.append(row)
    return rows


def _difficulty_from_math_level(level: int) -> str:
    return "Superhard" if int(level or 0) >= 5 else "Hard"


def _build_math500(existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing_sources = {row.get("_source_id") for row in existing}
    rows = _fetch_rows(MATH500_ROWS_URL, total=500)
    by_subject: Dict[str, List[Dict[str, Any]]] = {subject: [] for subject in MATH_SUBJECT_ORDER}
    for row in rows:
        subject = row.get("subject", "")
        if subject not in by_subject:
            continue
        if row.get("unique_id") in existing_sources:
            continue
        if int(row.get("level") or 0) < 4:
            continue
        by_subject[subject].append(row)

    additions: List[Dict[str, Any]] = []
    for subject in MATH_SUBJECT_ORDER:
        picks = by_subject[subject][:2]
        if len(picks) < 2:
            raise RuntimeError(f"Could not find two new MATH-500 level 4/5 seeds for {subject}.")
        prefix = SUBJECT_PREFIX[subject]
        for offset, row in enumerate(picks, start=3):
            additions.append(
                {
                    "ID": f"math500_{prefix}_{offset:02d}",
                    "Question": row["problem"],
                    "Answer": str(row["answer"]),
                    "Solution": row.get("solution", ""),
                    "Difficulty": _difficulty_from_math_level(int(row.get("level") or 0)),
                    "code": _print_answer_code(row["answer"]),
                    "_source_benchmark": "math500",
                    "_source_dataset": "HuggingFaceH4/MATH-500",
                    "_source_config": "default",
                    "_source_split": "test",
                    "_source_id": row.get("unique_id", ""),
                    "_source_row_idx": row.get("_row_idx"),
                    "_source_level": row.get("level"),
                    "_source_subject": subject,
                    "_ablation_seed_split": "new_diagnostic",
                }
            )
    return existing + additions


_GSM_FINAL_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")


def _gsm_answer(answer_text: str) -> str:
    match = _GSM_FINAL_RE.search(answer_text.replace(",", ""))
    if match:
        return match.group(1)
    return answer_text.split("####")[-1].strip().replace(",", "")


def _build_gsm8k(existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing_sources = {str(row.get("_source_id")) for row in existing}
    rows = _fetch_rows(GSM8K_ROWS_URL, total=80)
    additions: List[Dict[str, Any]] = []
    for row in rows:
        source_id = str(row.get("_row_idx"))
        if source_id in existing_sources:
            continue
        answer = _gsm_answer(str(row.get("answer", "")))
        additions.append(
            {
                "ID": f"gsm8k_new_{len(additions) + 1:02d}_{source_id}",
                "Question": row["question"],
                "Answer": answer,
                "Solution": row.get("answer", ""),
                "Difficulty": "Easy",
                "code": _print_answer_code(answer),
                "_source_benchmark": "gsm8k",
                "_source_dataset": "openai/gsm8k",
                "_source_config": "main",
                "_source_split": "test",
                "_source_id": source_id,
                "_source_row_idx": row.get("_row_idx"),
                "_ablation_seed_split": "new_diagnostic",
            }
        )
        if len(additions) >= 10:
            break
    if len(additions) < 10:
        raise RuntimeError("Could not find ten new GSM8K seeds.")
    return existing + additions


def _iter_aime_rows() -> Iterable[Dict[str, Any]]:
    for exam, url in AIME2025_URLS.items():
        for idx, line in enumerate(_get_text(url).splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_exam"] = exam
            row["_problem_number"] = idx
            yield row


def _build_aime2025(existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    additions: List[Dict[str, Any]] = []
    for row in _iter_aime_rows():
        # Keep the original AIME-I-heavy existing seed set separate. The new
        # expansion uses AIME II for provenance clarity and overlap avoidance.
        if row["_exam"] != "II":
            continue
        answer = str(row["answer"])
        additions.append(
            {
                "ID": f"aime2025_ii_p{row['_problem_number']:02d}",
                "Question": row["question"],
                "Answer": answer,
                "Solution": "",
                "Difficulty": "Superhard",
                "code": _print_answer_code(answer),
                "_source_benchmark": "aime2025",
                "_source_dataset": "opencompass/AIME2025",
                "_source_config": "AIME2025-II",
                "_source_split": "test",
                "_source_id": f"II-{row['_problem_number']}",
                "_source_problem_number": row["_problem_number"],
                "_ablation_seed_split": "new_diagnostic",
            }
        )
        if len(additions) >= 10:
            break
    if len(additions) < 10:
        raise RuntimeError("Could not find ten new AIME 2025 seeds.")
    return existing + additions


def _math_ids(rows: List[Dict[str, Any]], suffix: str) -> List[str]:
    return [row["ID"] for row in rows if row["ID"].endswith(suffix)]


def _new_ids(rows: List[Dict[str, Any]]) -> List[str]:
    return [row["ID"] for row in rows if row.get("_ablation_seed_split") == "new_diagnostic"]


def _manifest_for_benchmark(bench: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [row["ID"] for row in rows]
    existing = [row["ID"] for row in rows if row.get("_ablation_seed_split") == "existing_matched"]
    new = _new_ids(rows)
    if len(existing) != 10 or len(new) != 10:
        raise RuntimeError(f"{bench} expected 10 existing and 10 new seeds; got {len(existing)} and {len(new)}.")

    entry: Dict[str, Any] = {
        "seed_file": f"data/seed/external_ablation/{bench}_ablation20_seeds.json",
        "existing_all": {
            "description": "All existing matched seeds from the paper artifact.",
            "split": "existing_matched",
            "target_treatment_role": "matched_ablation",
            "seed_ids": existing,
        },
        "new_all": {
            "description": "All new diagnostic seeds for the 100-treatment expansion campaign.",
            "split": "new_diagnostic",
            "target_treatment_role": "new_seed_expansion",
            "seed_ids": new,
        },
    }

    if bench == "math500":
        entry.update(
            {
                "existing_a": {
                    "description": "Existing first seed of each MATH subject.",
                    "split": "existing_matched",
                    "target_treatment_role": "matched_ablation",
                    "seed_ids": _math_ids(rows, "_01"),
                },
                "existing_b": {
                    "description": "Existing second seed of each MATH subject.",
                    "split": "existing_matched",
                    "target_treatment_role": "matched_ablation",
                    "seed_ids": _math_ids(rows, "_02"),
                },
                "new_a": {
                    "description": "New first seed of each MATH subject.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": _math_ids(rows, "_03"),
                },
                "new_b": {
                    "description": "New second seed of each MATH subject.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": _math_ids(rows, "_04"),
                },
                "new_c": {
                    "description": "Cross-index mix over new MATH seeds.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": [new[i] for i in [0, 3, 4, 7, 8]],
                },
                "new_d": {
                    "description": "Complementary cross-index mix over new MATH seeds.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": [new[i] for i in [1, 2, 5, 6, 9]],
                },
            }
        )
    elif bench == "aime2025":
        entry.update(
            {
                "existing_a": {
                    "description": "Existing AIME combo matching the paper artifact combo_a.",
                    "split": "existing_matched",
                    "target_treatment_role": "matched_ablation",
                    "seed_ids": [
                        "aime2025_alge_p04",
                        "aime2025_alge_p08",
                        "aime2025_comb_p03",
                        "aime2025_geom_p02",
                        "aime2025_numb_p01",
                    ],
                },
                "existing_b": {
                    "description": "Existing AIME combo matching the paper artifact combo_b.",
                    "split": "existing_matched",
                    "target_treatment_role": "matched_ablation",
                    "seed_ids": [
                        "aime2025_alge_p09",
                        "aime2025_comb_p05",
                        "aime2025_comb_p07",
                        "aime2025_geom_p06",
                        "aime2025_numb_p15",
                    ],
                },
                "new_a": {
                    "description": "New first five seeds.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": new[:5],
                },
                "new_b": {
                    "description": "New second five seeds.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": new[5:],
                },
                "new_c": {
                    "description": "Cross-index mix over new seeds.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": [new[i] for i in [0, 2, 5, 7, 9]],
                },
                "new_d": {
                    "description": "Complementary cross-index mix over new seeds.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": [new[i] for i in [1, 3, 4, 6, 8]],
                },
            }
        )
    else:
        entry.update(
            {
                "existing_a": {
                    "description": "Existing first five seeds.",
                    "split": "existing_matched",
                    "target_treatment_role": "matched_ablation",
                    "seed_ids": existing[:5],
                },
                "existing_b": {
                    "description": "Existing second five seeds.",
                    "split": "existing_matched",
                    "target_treatment_role": "matched_ablation",
                    "seed_ids": existing[5:],
                },
                "new_a": {
                    "description": "New first five seeds.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": new[:5],
                },
                "new_b": {
                    "description": "New second five seeds.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": new[5:],
                },
                "new_c": {
                    "description": "Cross-index mix over new seeds.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": [new[i] for i in [0, 2, 5, 7, 9]],
                },
                "new_d": {
                    "description": "Complementary cross-index mix over new seeds.",
                    "split": "new_diagnostic",
                    "target_treatment_role": "new_seed_expansion",
                    "seed_ids": [new[i] for i in [1, 3, 4, 6, 8]],
                },
            }
        )
    for combo_name, combo in entry.items():
        if combo_name == "seed_file":
            continue
        if len(combo["seed_ids"]) not in {5, 10}:
            raise RuntimeError(f"{bench}.{combo_name} has {len(combo['seed_ids'])} seeds: {combo['seed_ids']}")
        missing = [seed_id for seed_id in combo["seed_ids"] if seed_id not in ids]
        if missing:
            raise RuntimeError(f"{bench}.{combo_name} references unknown seeds: {missing}")
    return entry


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks = {
        "math500": _build_math500(_load_existing("math500_seeds.json")),
        "gsm8k": _build_gsm8k(_load_existing("gsm8k_seeds.json")),
        "aime2025": _build_aime2025(_load_existing("aime2025_seeds.json")),
    }
    manifest: Dict[str, Any] = {}
    for bench, rows in benchmarks.items():
        _write_json(OUT_DIR / f"{bench}_ablation20_seeds.json", rows)
        manifest[bench] = _manifest_for_benchmark(bench, rows)
    (OUT_DIR / "combo_manifest_ablation.yaml").write_text(
        "# EntropyMath external ablation/new-seed expansion combo manifest\n"
        + yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(json.dumps({bench: len(rows) for bench, rows in benchmarks.items()}, indent=2))
    print(OUT_DIR / "combo_manifest_ablation.yaml")


if __name__ == "__main__":
    main()
