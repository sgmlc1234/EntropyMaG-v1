"""Runtime parameters, generation sizes, and stop-condition helpers extracted from graph_full.py."""

from typing import Any, Dict, Optional
from config import DEFAULT_PARAMS, STEADY_STATE_POOL_SIZE
from data_paths import DEFAULT_GENERATION_FORMAT
from deepagent.state_full import AgentState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIFFICULTY_LABEL_TO_SCORE = {
    "easy": 4.5,
    "medium": 6.5,
    "hard": 8.5,
    "superhard": 9.5,
}

MAX_PARALLEL_DISPATCH_FALLBACK = 4
LINEAGE_DEPTH_EASY_THRESHOLD = 3
CROSSOVER_DEPTH_EASY_THRESHOLD = 2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_difficulty_value(value, default: float = 7.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return default
    if text in DIFFICULTY_LABEL_TO_SCORE:
        return DIFFICULTY_LABEL_TO_SCORE[text]
    try:
        return float(text)
    except Exception:
        return default


def _canonical_target_generation_size(parameters: Dict[str, Any]) -> int:
    return STEADY_STATE_POOL_SIZE


def _desired_generation_size(state: AgentState, params: Optional[Dict[str, Any]] = None) -> int:
    desired = state.get("desired_generation_size")
    if desired is not None:
        return max(1, int(desired))
    return _canonical_target_generation_size(params if params is not None else dict(state.get("parameters", {}) or {}))


def _current_generation_size(state: AgentState) -> int:
    return int(state.get("current_generation_size") or len(state.get("current_generation", []) or []))


def _cumulative_validated_generated_count(state: AgentState) -> int:
    return max(0, int(state.get("cumulative_validated_generated_count", 0) or 0))


def _target_problem_count(state: AgentState) -> Optional[int]:
    value = (state.get("session_options", {}) or {}).get("target_problem_count")
    if value in {None, "", 0}:
        return None
    return max(1, int(value))


def _completion_stop_reason(state: AgentState) -> str:
    options = state.get("session_options", {}) or {}
    max_generations = options.get("max_generations")
    gen_count = int(state.get("generation_count", 0) or 0)
    if max_generations is not None and gen_count >= int(max_generations):
        return "max_generations"
    target_problem_count = _target_problem_count(state)
    if target_problem_count is not None and _cumulative_validated_generated_count(state) >= target_problem_count:
        return "target_problem_count"
    return ""


def _normalized_parameters(state: AgentState) -> Dict:
    params = dict(DEFAULT_PARAMS)
    params.update(state.get("parameters", {}) or {})
    gen_count = state.get("generation_count", 0)
    params["target_generation_size"] = _canonical_target_generation_size(params)
    params["population_size"] = params["target_generation_size"]
    if gen_count <= 0:
        params["easy_ratio"] = 0.8
    elif gen_count == 1:
        params["easy_ratio"] = 0.7
    elif gen_count == 2:
        params["easy_ratio"] = 0.2
    else:
        params["easy_ratio"] = 0.1

    options = state.get("session_options", {}) or {}
    if options.get("mutation_only"):
        params["mutation_only"] = True
    params.setdefault("require_all_valid", True)
    params.setdefault("auto_continue_generations", True)
    save_format = options.get("save_format")
    if save_format:
        params["save_format"] = save_format
    else:
        params["save_format"] = DEFAULT_GENERATION_FORMAT
    params["gen_count"] = gen_count
    return params


def _build_review_prompt(state: AgentState) -> str:
    gen_count = state.get("generation_count", 0)
    candidates = state.get("current_generation", [])
    valid_count = len(candidates)
    save_status = state.get("generation_save_status", "") or ""
    stop_reason = _completion_stop_reason(state)
    if stop_reason == "max_generations":
        prefix = "partially saved" if save_status == "partial_save" else "ready"
        return f"Generation {gen_count} is {prefix} ({valid_count}/{len(candidates)} valid). Max generations reached. Options: [advisor, modify, exit]"
    if stop_reason == "target_problem_count":
        return (
            f"Generation {gen_count} is {'partially saved' if save_status == 'partial_save' else 'ready'} ({valid_count}/{len(candidates)} valid). "
            f"Target problem count reached ({_cumulative_validated_generated_count(state)}/{_target_problem_count(state)}). "
            "Options: [advisor, modify, exit]"
        )
    if save_status == "partial_save":
        return f"Generation {gen_count} partially saved ({valid_count}/{_canonical_target_generation_size(state.get('parameters', {}) or {})} target). Auto-continuing to recover pool size next generation."
    return f"Generation {gen_count} saved ({valid_count}/{len(candidates)} valid). Auto-continuing to the next generation."
