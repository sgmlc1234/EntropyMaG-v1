import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_GENERATION_FORMAT = "data/gen_problem/generation_deep_{gen}.json"
_GENERATION_NUMBER_RE = re.compile(r"generation(?:_deep)?_(\d+)\.json$")
_EPHEMERAL_RUN_DIR_PREFIXES = ("deep-trace-", "deep-trace-smoke-", "deep-trace-rot-", "deep-trace-sbx-")


def normalize_problem(problem: Dict) -> Dict:
    normalized = dict(problem)
    normalized.setdefault("id", normalized.get("ID") or normalized.get("Id"))
    normalized.setdefault("statement", normalized.get("Question") or normalized.get("question"))
    if "answer" not in normalized and "Answer" in normalized:
        normalized["answer"] = str(normalized["Answer"])
    if "solution" not in normalized and "Solution" in normalized:
        normalized["solution"] = normalized["Solution"]
    if "difficulty" not in normalized and "Difficulty" in normalized:
        normalized["difficulty"] = normalized["Difficulty"]
    if "answer_type" not in normalized:
        answer = str(normalized.get("answer", "")).replace(",", "").strip()
        normalized["answer_type"] = "integer" if re.fullmatch(r"[-+]?\d+", answer) else "other"
    return normalized


def load_problem_file(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list of problems")
    return [normalize_problem(problem) for problem in raw]


def _discover_seed_files(pattern: str) -> List[str]:
    return sorted(
        path
        for path in glob.glob(pattern)
        if path.endswith(".json") and not path.endswith(".schema.json") and not path.endswith(".private.json")
    )


def public_seed_files() -> List[str]:
    return _discover_seed_files("data/seed/*.json")


def private_seed_files() -> List[str]:
    return _discover_seed_files("data/seed/private/*.json")


def resolve_seed_files(seed_spec: Optional[str] = None) -> List[str]:
    if seed_spec:
        files = [part.strip() for part in seed_spec.split(",") if part.strip()]
        if files:
            return files

    discovered_public = public_seed_files()
    if discovered_public:
        return [discovered_public[0]]

    discovered_private = private_seed_files()
    if discovered_private:
        return discovered_private

    raise FileNotFoundError(
        "No seed files found under data/seed. "
        "In a public release, provide --seed-file explicitly or add a public seed JSON under data/seed/."
    )


def load_seed_problems(seed_spec: Optional[str] = None) -> List[Dict]:
    problems: List[Dict] = []
    for path in resolve_seed_files(seed_spec):
        problems.extend(load_problem_file(path))
    if not problems:
        raise ValueError("Seed files were found but contained no problems")
    return problems


def extract_generation_number(path: str) -> int:
    match = _GENERATION_NUMBER_RE.search(os.path.basename(path))
    if match:
        return int(match.group(1))
    return -1


def is_ephemeral_generation_path(path: str) -> bool:
    normalized = os.path.normpath(path)
    parts = normalized.split(os.sep)
    return any(part.startswith(prefix) for part in parts for prefix in _EPHEMERAL_RUN_DIR_PREFIXES)


def list_generation_files(save_format: Optional[str] = None) -> List[str]:
    patterns: List[str] = []
    if save_format:
        if "{gen}" in save_format:
            patterns.append(save_format.replace("{gen}", "*"))
        else:
            patterns.append(save_format)

    patterns.extend(
        [
            "data/gen_problem/**/*.json",
            "data/generation_deep_*.json",
            "data/generation_*.json",
        ]
    )

    files = set()
    for pattern in patterns:
        files.update(glob.glob(pattern, recursive=True))

    generation_files = [
        path
        for path in files
        if "generation" in os.path.basename(path) and not is_ephemeral_generation_path(path)
    ]
    return sorted(generation_files)


def load_latest_generation(save_format: Optional[str] = None) -> Tuple[Optional[List[Dict]], int, Optional[str]]:
    files = list_generation_files(save_format=save_format)
    if not files:
        return None, 0, None

    ranked = sorted(
        files,
        key=lambda path: (
            extract_generation_number(path),
            os.path.getmtime(path),
            path,
        ),
        reverse=True,
    )
    for candidate in ranked:
        try:
            return load_problem_file(candidate), max(extract_generation_number(candidate), 0), candidate
        except (json.JSONDecodeError, ValueError):
            continue

    return None, 0, None


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
