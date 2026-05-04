import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_paths import extract_generation_number, load_problem_file, normalize_problem

MODULE_DIR = Path(__file__).resolve().parent


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _validation_snapshot_paths(base_dir: Path = MODULE_DIR) -> List[Path]:
    return sorted(base_dir.glob("*_validate_candidates.json"))


def _generation_paths(base_dir: Path = MODULE_DIR) -> List[Path]:
    return sorted(
        base_dir.glob("generation_*.json"),
        key=lambda path: (extract_generation_number(str(path)), path.name),
    )


def _latest_candidate_from_generation(base_dir: Path = MODULE_DIR) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for generation_path in reversed(_generation_paths(base_dir)):
        generation_number = extract_generation_number(str(generation_path))
        try:
            problems = load_problem_file(str(generation_path))
        except Exception:
            continue
        if not problems:
            continue
        for slot, raw_problem in reversed(list(enumerate(problems))):
            problem = normalize_problem(raw_problem)
            if problem.get("type") in {"survivor", "fallback_survivor"}:
                continue
            return problem, {
                "source": "generation_file",
                "generation_count": generation_number,
                "slot": slot,
                "generation_file": generation_path.name,
            }
        problem = normalize_problem(problems[-1])
        return problem, {
            "source": "generation_file",
            "generation_count": generation_number,
            "slot": len(problems) - 1,
            "generation_file": generation_path.name,
        }
    return None


def _validated_candidates_from_generation(base_dir: Path = MODULE_DIR) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    items: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen_ids = set()
    for generation_path in _generation_paths(base_dir):
        generation_number = extract_generation_number(str(generation_path))
        try:
            problems = load_problem_file(str(generation_path))
        except Exception:
            continue
        for slot, raw_problem in enumerate(problems):
            problem = normalize_problem(raw_problem)
            problem_id = problem.get("id")
            if not problem_id or problem.get("type") in {"survivor", "fallback_survivor"} or problem_id in seen_ids:
                continue
            items.append(
                (
                    problem,
                    {
                        "source": "generation_file",
                        "generation_count": generation_number,
                        "slot": slot,
                        "generation_file": generation_path.name,
                    },
                )
            )
            seen_ids.add(problem_id)
    return items


def _candidate_from_worker_chain(base_dir: Path = MODULE_DIR) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    payload = _read_json(base_dir / "worker_chain_smoke.json")
    if not payload:
        return None
    problem = payload.get("generated_problem") or {}
    validation = payload.get("validation") or {}
    if not problem:
        return None
    meta = {
        "source": "worker_chain_smoke",
        "seed_id": payload.get("seed_id"),
        "validation": validation,
        "research_artifact": payload.get("research_artifact"),
    }
    return problem, meta


def _validated_candidates_from_worker_chain(base_dir: Path = MODULE_DIR) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    payload = _read_json(base_dir / "worker_chain_smoke.json")
    if not payload:
        return []
    problem = payload.get("generated_problem") or {}
    validation = payload.get("validation") or {}
    if not problem or not validation.get("valid"):
        return []
    meta = {
        "source": "worker_chain_smoke",
        "seed_id": payload.get("seed_id"),
        "validation": validation,
        "research_artifact": payload.get("research_artifact"),
    }
    return [(problem, meta)]


def _candidate_from_validation_snapshot(base_dir: Path = MODULE_DIR) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    chosen = None
    chosen_meta = None

    def _normalized_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip()).lower()

    def _is_too_similar(problem: Dict[str, Any]) -> bool:
        statement = _normalized_text(problem.get("statement", ""))
        parent_ids = problem.get("parent_ids", []) or []
        if len(parent_ids) != 1:
            return False
        for seed_payload_path in sorted(base_dir.glob("*_plan_generation.json")):
            seed_payload = _read_json(seed_payload_path)
            if not seed_payload:
                continue
            for item in seed_payload.get("work_items", []) or []:
                for parent in item.get("parents", []) or []:
                    if parent.get("id") == parent_ids[0]:
                        parent_statement = _normalized_text(parent.get("statement", ""))
                        if parent_statement and statement:
                            if statement == parent_statement:
                                return True
                            if statement in parent_statement or parent_statement in statement:
                                return True
        return False

    for snapshot_path in reversed(_validation_snapshot_paths(base_dir)):
        payload = _read_json(snapshot_path)
        if not payload:
            continue
        approved = payload.get("approved_candidates") or []
        if not approved:
            continue
        for item in approved:
            problem = item.get("problem") or {}
            if problem.get("type") not in {"survivor", "fallback_survivor"} and not _is_too_similar(problem):
                chosen = problem
                chosen_meta = {
                    "source": "validate_snapshot",
                    "generation_count": payload.get("generation_count"),
                    "valid_count": payload.get("valid_count"),
                    "validation_feedback": payload.get("validation_feedback", []),
                }
                break
        if chosen is None:
            for item in approved:
                problem = item.get("problem") or {}
                if problem.get("type") not in {"survivor", "fallback_survivor"}:
                    chosen = problem
                    chosen_meta = {
                        "source": "validate_snapshot",
                        "generation_count": payload.get("generation_count"),
                        "valid_count": payload.get("valid_count"),
                        "validation_feedback": payload.get("validation_feedback", []),
                    }
                    break
        if chosen is not None:
            break
    if not chosen:
        return None
    return chosen, chosen_meta or {}


def _validated_candidates_from_snapshot(base_dir: Path = MODULE_DIR) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    items: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen_ids = set()
    for snapshot_path in _validation_snapshot_paths(base_dir):
        payload = _read_json(snapshot_path)
        if not payload:
            continue
        approved = payload.get("approved_candidates") or []
        for entry in approved:
            problem = entry.get("problem") or {}
            problem_id = problem.get("id")
            if not problem or problem.get("type") in {"survivor", "fallback_survivor"} or not problem_id or problem_id in seen_ids:
                continue
            meta = {
                "source": "validate_snapshot",
                "generation_count": payload.get("generation_count"),
                "valid_count": payload.get("valid_count"),
                "slot": entry.get("slot"),
                "validation_feedback": payload.get("validation_feedback", []),
            }
            items.append((problem, meta))
            seen_ids.add(problem_id)
    return items


def _latest_candidate(base_dir: Path = MODULE_DIR) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    return (
        _latest_candidate_from_generation(base_dir)
        or _candidate_from_worker_chain(base_dir)
        or _candidate_from_validation_snapshot(base_dir)
    )


def _all_validated_candidates(base_dir: Path = MODULE_DIR) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    generation_items = _validated_candidates_from_generation(base_dir)
    if generation_items:
        return generation_items
    worker_items = _validated_candidates_from_worker_chain(base_dir)
    if worker_items:
        return worker_items
    return _validated_candidates_from_snapshot(base_dir)


def _render_statement_markdown(statement: str) -> str:
    text = (statement or "").strip()
    if not text:
        return ""
    if "$" in text or "\\[" in text or "\\begin{" in text:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        sentence = lines[0]
        replacements = [
            r"a\s*\+\s*b\s*\+\s*c\s*=\s*151",
            r"a\^2\s*\+\s*b\^2\s*\+\s*c\^2\s*=\s*10939",
            r"a\^3\s*\+\s*b\^3\s*\+\s*c\^3\s*=\s*957871",
            r"a\^4\s*\+\s*b\^4\s*\+\s*c\^4",
            r"\|N\s*-\s*p\|",
        ]
        for pattern in replacements:
            sentence = re.sub(
                pattern,
                lambda match: f"${match.group(0).replace(' ', '')}$",
                sentence,
            )
        return sentence
    return "\n\n".join(lines)


def _render_problem_markdown(problem: Dict[str, Any], meta: Dict[str, Any]) -> str:
    statement = _render_statement_markdown(problem.get("statement", ""))
    research = problem.get("research_artifact") or meta.get("research_artifact") or {}
    lines = [
        "# Latest Generated Problem",
        "",
        "## Metadata",
        f"- `source`: {meta.get('source', 'unknown')}",
        f"- `problem_id`: {problem.get('id', 'unknown')}",
        f"- `type`: {problem.get('type', 'unknown')}",
        f"- `difficulty`: {problem.get('difficulty', 'unknown')}",
        f"- `difficulty_label`: {problem.get('difficulty_label', 'unknown')}",
    ]
    if meta.get("seed_id"):
        lines.append(f"- `seed_id`: {meta.get('seed_id')}")
    validation = meta.get("validation") or {}
    if validation:
        lines.append(f"- `validation_valid`: {validation.get('valid')}")
        lines.append(f"- `validation_reason`: {validation.get('reason')}")
    if research:
        lines.append(f"- `research_tool`: {research.get('tool_used', 'unknown')}")
        lines.append(f"- `research_source_count`: {research.get('source_count', 0)}")
    lines.extend(
        [
            "",
            "## Statement",
            "",
            statement,
            "",
            "## Answer",
            "",
            str(problem.get("answer", "")),
            "",
            "## Solution",
            "",
            problem.get("solution", ""),
            "",
            "## Verification Code",
            "",
            "```python",
            problem.get("code", ""),
            "```",
        ]
    )
    if research:
        lines.extend(
            [
                "",
                "## Research Artifact",
                "",
                f"- `query`: {research.get('query', '')}",
                f"- `tool_used`: {research.get('tool_used', '')}",
                f"- `source_count`: {research.get('source_count', 0)}",
                "",
                "### Synthesis",
                "",
                research.get("short_synthesis", ""),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _safe_problem_slug(problem: Dict[str, Any], index: int) -> str:
    """Filesystem-safe slug from a problem id.

    Caps total slug length at 80 chars to avoid macOS/Linux 255-byte
    filename limits when the lineage-readable fallback id chains through
    many generations of synth-seed reuse (e.g.
    "mut_easy_cross_easy_mut_hard_..._AC2" growing past 250 chars and
    crashing the post-run write_all_validated_problem_views step).
    When truncation is needed, append an 8-char hash of the full id so
    the slug remains unique while staying within filesystem limits.
    """
    pid = problem.get("id") or f"problem_{index:03d}"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", pid).strip("_")
    if not slug:
        return f"problem_{index:03d}"
    MAX_SLUG_LEN = 80
    if len(slug) > MAX_SLUG_LEN:
        import hashlib
        digest = hashlib.sha1(pid.encode("utf-8")).hexdigest()[:8]
        # Keep readable head + tail to preserve some visible lineage info
        head = slug[:30]
        tail = slug[-30:]
        slug = f"{head}__{digest}__{tail}"
    return slug


def write_all_validated_problem_views(base_dir: Path = MODULE_DIR) -> List[Path]:
    candidates = _all_validated_candidates(base_dir)
    if not candidates:
        (base_dir / "ALL_VALIDATED_PROBLEMS.json").write_text(
            json.dumps({"count": 0, "problems": []}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (base_dir / "ALL_VALIDATED_PROBLEMS.md").write_text(
            "# All Validated Problems\n\nNo validated generated problems were found in this run.\n",
            encoding="utf-8",
        )
        return []
    validated_dir = base_dir / "validated_problems"
    validated_dir.mkdir(parents=True, exist_ok=True)
    for stale in validated_dir.glob("*"):
        if stale.is_file():
            stale.unlink()
    index_payload = []
    written: List[Path] = []
    for idx, (problem, meta) in enumerate(candidates, 1):
        slug = _safe_problem_slug(problem, idx)
        json_path = validated_dir / f"{idx:03d}_{slug}.json"
        md_path = validated_dir / f"{idx:03d}_{slug}.md"
        payload = {"meta": meta, "problem": problem}
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(_render_problem_markdown(problem, meta), encoding="utf-8")
        index_payload.append(
            {
                "index": idx,
                "problem_id": problem.get("id", ""),
                "type": problem.get("type", ""),
                "difficulty": problem.get("difficulty", ""),
                "difficulty_label": problem.get("difficulty_label", ""),
                "json_file": str(json_path.relative_to(base_dir)),
                "markdown_file": str(md_path.relative_to(base_dir)),
            }
        )
        written.append(md_path)

    (base_dir / "ALL_VALIDATED_PROBLEMS.json").write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_lines = ["# All Validated Generated Problems", ""]
    for item in index_payload:
        md_lines.extend(
            [
                f"## {item['index']}. {item['problem_id']}",
                f"- `type`: {item['type']}",
                f"- `difficulty`: {item['difficulty']}",
                f"- `difficulty_label`: {item['difficulty_label']}",
                f"- `json`: `{item['json_file']}`",
                f"- `markdown`: `{item['markdown_file']}`",
                "",
            ]
        )
    (base_dir / "ALL_VALIDATED_PROBLEMS.md").write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    return written


def write_latest_problem_views(base_dir: Path = MODULE_DIR) -> Optional[Path]:
    candidate = _latest_candidate(base_dir)
    if candidate is None:
        return None
    problem, meta = candidate
    (base_dir / "LATEST_GENERATED_PROBLEM.json").write_text(
        json.dumps({"meta": meta, "problem": problem}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path = base_dir / "LATEST_GENERATED_PROBLEM.md"
    md_path.write_text(_render_problem_markdown(problem, meta), encoding="utf-8")
    return md_path


__all__ = ["write_all_validated_problem_views", "write_latest_problem_views"]
