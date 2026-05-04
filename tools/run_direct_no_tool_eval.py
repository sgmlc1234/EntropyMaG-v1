#!/usr/bin/env python3
"""Run direct no-tool model evaluation over a JSONL math dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_TOKENS = 8192
SYSTEM_PROMPT = (
    "You are solving a mathematics problem. Work independently without tools. "
    "Give the final answer in exactly one \\boxed{...} expression."
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_answer(value: object) -> str:
    text = str(value or "").strip()
    text = text.replace("\\,", "")
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = text.strip().strip("$").strip()
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _to_float(value: str) -> Optional[float]:
    text = _normalize_answer(value)
    if not text:
        return None
    try:
        return float(Fraction(text))
    except Exception:
        pass
    try:
        return float(Decimal(text))
    except (InvalidOperation, ValueError):
        return None


def answers_equal(predicted: object, gold: object, tol: float = 1e-6) -> bool:
    pred_norm = _normalize_answer(predicted)
    gold_norm = _normalize_answer(gold)
    if pred_norm and pred_norm == gold_norm:
        return True
    pred_float = _to_float(str(predicted))
    gold_float = _to_float(str(gold))
    if pred_float is not None and gold_float is not None:
        return abs(pred_float - gold_float) <= tol
    return False


def extract_boxed(text: str) -> str:
    marker = "\\boxed{"
    start = str(text or "").rfind(marker)
    if start < 0:
        return ""
    idx = start + len(marker)
    depth = 1
    chars: List[str] = []
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(char)
        else:
            chars.append(char)
        idx += 1
    return ""


def call_model(client: OpenAI, model: str, question: str, max_tokens: int) -> tuple[str, Dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
    return content, {"id": response.id, "usage": usage}


def result_path(output_dir: Path, idx: int, repeat: int) -> Path:
    return output_dir / f"{idx}_run_{repeat}.json"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = args.dataset_jsonl if args.dataset_jsonl.is_absolute() else REPO / args.dataset_jsonl
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    rows = _read_jsonl(dataset_path)
    if args.limit:
        rows = rows[: args.limit]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENROUTER_API_KEY is not set; use --dry-run to validate the job without API calls.")

    client = None
    if not args.dry_run:
        client = OpenAI(api_key=api_key, base_url=args.base_url)

    completed = 0
    correct = 0
    extraction_fail = 0
    for idx, row in enumerate(rows):
        for repeat in range(args.repeats):
            out_path = result_path(output_dir, idx, repeat)
            if args.resume and out_path.exists():
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                completed += 1
                correct += int(bool(payload.get("correct")))
                extraction_fail += int(not payload.get("final_answer"))
                continue

            if args.dry_run:
                payload = {
                    "dry_run": True,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_jsonl": str(dataset_path.relative_to(REPO) if dataset_path.is_relative_to(REPO) else dataset_path),
                    "row_index": idx,
                    "repeat": repeat,
                    "model": args.model,
                    "problem_id": row.get("id", ""),
                    "benchmark": row.get("_source_benchmark", ""),
                    "arm": args.eval_arm or row.get("_arm", ""),
                    "gold_answer": row.get("answer", ""),
                    "final_answer": "",
                    "correct": False,
                    "solver_protocol": "direct_no_tool",
                }
                _write_json(out_path, payload)
                completed += 1
                continue

            assert client is not None
            started = time.time()
            content, response_meta = call_model(client, args.model, row.get("question", ""), args.max_tokens)
            elapsed = time.time() - started
            final_answer = extract_boxed(content)
            is_correct = answers_equal(final_answer, row.get("answer", ""))
            payload = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_jsonl": str(dataset_path.relative_to(REPO) if dataset_path.is_relative_to(REPO) else dataset_path),
                "row_index": idx,
                "repeat": repeat,
                "model": args.model,
                "problem_id": row.get("id", ""),
                "benchmark": row.get("_source_benchmark", ""),
                "arm": args.eval_arm or row.get("_arm", ""),
                "ablation_seed_split": row.get("_ablation_seed_split", ""),
                "parent_ids": row.get("_parent_ids", []),
                "gold_answer": row.get("answer", ""),
                "final_answer": final_answer,
                "correct": is_correct,
                "answer_extraction_status": "boxed_found" if final_answer else "boxed_missing",
                "elapsed_time_sec": elapsed,
                "history": [{"role": "assistant", "content": content}],
                "response_meta": response_meta,
                "solver_protocol": "direct_no_tool",
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "parser_version": "boxed_extractor_v2",
            }
            _write_json(out_path, payload)
            completed += 1
            correct += int(is_correct)
            extraction_fail += int(not final_answer)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_jsonl": str(dataset_path.relative_to(REPO) if dataset_path.is_relative_to(REPO) else dataset_path),
        "output_dir": str(output_dir.relative_to(REPO) if output_dir.is_relative_to(REPO) else output_dir),
        "model": args.model,
        "rows": len(rows),
        "repeats": args.repeats,
        "completed_runs": completed,
        "correct_runs": correct,
        "accuracy": correct / completed if completed else 0.0,
        "extraction_fail_runs": extraction_fail,
        "dry_run": bool(args.dry_run),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-arm", default="", help="Analysis arm label to store in result payloads.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
