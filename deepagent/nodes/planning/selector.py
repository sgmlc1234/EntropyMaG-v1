"""DeepAgent selector for orchestrator-owned generation planning."""
import hashlib
import logging
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith.run_helpers import get_current_run_tree, tracing_context

from config import STEADY_STATE_POOL_SIZE, get_llm_config
from deepagent.tracing import _is_retryable_structured_error, record_deterministic_span
from prompts import (
    SelectorPlanSchema,
    SELECTOR_SYSTEM_PROMPT,
    build_op_type_allocation_block,
    build_run_working_memory_block,
    build_selector_prompt,
)

logger = logging.getLogger("deep_selector")

config = get_llm_config("selector")
llm = ChatOpenAI(
    model=config["model"],
    temperature=0.1,
    max_tokens=config.get("max_tokens", 1000),
    timeout=config.get("timeout", 120),
    max_retries=config.get("max_retries", 3),
    api_key=config["api_key"],
    base_url=config["base_url"],
)

LAST_SELECTION = None


def _slim_parent(problem: Dict) -> Dict:
    return {
        "id": problem.get("id", ""),
        "statement": problem.get("statement", ""),
        "answer": str(problem.get("answer", "")),
        "difficulty": problem.get("difficulty", ""),
        "difficulty_label": problem.get("difficulty_label", ""),
        "type": problem.get("type", ""),
        "family_signature": problem.get("family_signature", ""),
        "parent_ids": list(problem.get("parent_ids", []) or []),
        "lineage_metrics": dict(problem.get("lineage_metrics", {}) or {}),
    }


def _canonical_target_generation_size(parameters: Dict) -> int:
    return STEADY_STATE_POOL_SIZE


def _normalize_target_diff(value: float, difficulty_label: str, mode: str) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = 0.0
    if parsed <= 1.0:
        if difficulty_label in {"hard", "superhard"} or mode == "hard":
            return 9.0
        return 6.0
    return max(1.0, min(10.0, parsed))


def _gen_summary(current_generation: List[Dict], invariant_bundles: Optional[Dict[str, Dict]]) -> str:
    lines = []
    for problem in current_generation:
        bundle = (invariant_bundles or {}).get(problem.get("id"), {})
        axes = ", ".join(bundle.get("allowed_variation_axes", [])[:3]) or "unspecified"
        lines.append(
            f"ID: {problem.get('id')}, Difficulty: {problem.get('difficulty', 'Unknown')}, "
            f"Type: {problem.get('type', 'Unknown')}, Allowed variation axes: {axes}"
        )
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    trimmed = text.strip()
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[: max_chars - 1].rstrip() + "\u2026"


def _seed_bodies_block(
    current_generation: List[Dict],
    invariant_bundles: Optional[Dict[str, Dict]],
    *,
    max_parents: int = 6,
    statement_chars: int = 600,
    solution_chars: int = 400,
) -> str:
    """Render a seed_bodies block with truncated statement/solution for each parent.

    The selector LLM uses this block to ground variation_axis, seed_focus, and
    query_hint in the actual parent content rather than just metadata summaries.
    """
    if not current_generation:
        return "Seed bodies: (none provided)"
    chunks: List[str] = []
    for problem in current_generation[:max_parents]:
        pid = problem.get("id", "")
        if not pid:
            continue
        bundle = (invariant_bundles or {}).get(pid, {})
        axes = ", ".join(bundle.get("allowed_variation_axes", [])[:3]) or "unspecified"
        statement = _truncate(problem.get("statement", ""), statement_chars) or "(missing)"
        solution = _truncate(problem.get("solution", ""), solution_chars) or "(missing)"
        answer = _truncate(str(problem.get("answer", "")), 80) or "(missing)"
        family = problem.get("family_signature", "") or "(unknown)"
        chunks.append(
            f"[{pid}] family={family} axes={axes}\n"
            f"  statement: {statement}\n"
            f"  answer: {answer}\n"
            f"  solution: {solution}"
        )
    body = "\n".join(chunks) if chunks else "(none provided)"
    truncated_note = ""
    if len(current_generation) > max_parents:
        truncated_note = f"\n(+{len(current_generation) - max_parents} additional parents omitted for prompt budget)"
    return "Seed bodies (authoritative parent content; ground every non-survivor slot in these):\n" + body + truncated_note


def _selector_trace_inputs(
    current_generation: List[Dict],
    invariant_bundles: Optional[Dict[str, Dict]],
    run_working_memory: Optional[Dict],
    *,
    current_generation_size: int,
    target_generation_size: int,
    desired_generation_size_hint: int,
    cumulative_validated_generated_count: int,
    target_problem_count: Optional[int],
    normalization_mode_hint: str,
    selector_feedback: str,
) -> Dict:
    seed_bodies = _seed_bodies_block(current_generation, invariant_bundles)
    seed_bodies_digest = hashlib.sha1(seed_bodies.encode("utf-8")).hexdigest()[:12]
    return {
        "generation_summary": _gen_summary(current_generation, invariant_bundles),
        "seed_bodies_digest": seed_bodies_digest,
        "seed_bodies_char_count": len(seed_bodies),
        "run_working_memory": build_run_working_memory_block(run_working_memory or {}, stage="selector"),
        "runtime_state": {
            "current_generation_size": current_generation_size,
            "target_generation_size": target_generation_size,
            "desired_generation_size_hint": desired_generation_size_hint,
            "cumulative_validated_generated_count": cumulative_validated_generated_count,
            "target_problem_count": target_problem_count,
            "normalization_mode_hint": normalization_mode_hint,
        },
        "selector_feedback": selector_feedback,
    }


def _selector_trace_outputs(plan: Dict, raw_message) -> Dict:
    response_metadata = getattr(raw_message, "response_metadata", {}) or {}
    token_usage = dict(response_metadata.get("token_usage", {}) or {})
    slim_token_usage = {
        "prompt_tokens": token_usage.get("prompt_tokens", 0),
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
        "cost": token_usage.get("cost"),
    }
    return {
        "plan": plan,
        "token_usage": slim_token_usage,
        "model_name": response_metadata.get("model_name", ""),
        "finish_reason": response_metadata.get("finish_reason", ""),
    }


def get_selection_strategy(
    current_generation: List[Dict],
    invariant_bundles: Optional[Dict[str, Dict]] = None,
    run_working_memory: Optional[Dict] = None,
    *,
    current_generation_size: int,
    target_generation_size: int,
    desired_generation_size_hint: int,
    cumulative_validated_generated_count: int,
    target_problem_count: Optional[int],
    normalization_mode_hint: str,
    selector_feedback: str = "",
    op_type_allocation: Optional[Dict] = None,
    recent_survivor_answers: Optional[List[str]] = None,
) -> Dict:
    prompt_content = build_selector_prompt(
        _gen_summary(current_generation, invariant_bundles),
        build_run_working_memory_block(run_working_memory or {}, stage="selector"),
        current_generation_size=current_generation_size,
        target_generation_size=target_generation_size,
        desired_generation_size_hint=desired_generation_size_hint,
        cumulative_validated_generated_count=cumulative_validated_generated_count,
        target_problem_count=target_problem_count,
        normalization_mode_hint=normalization_mode_hint,
        selector_feedback=selector_feedback,
        seed_bodies_block=_seed_bodies_block(current_generation, invariant_bundles),
        op_type_allocation_block=build_op_type_allocation_block(op_type_allocation),
        recent_survivor_answers=recent_survivor_answers,
    )
    messages = [SystemMessage(content=SELECTOR_SYSTEM_PROMPT), HumanMessage(content=prompt_content)]
    trace_inputs = _selector_trace_inputs(
        current_generation,
        invariant_bundles,
        run_working_memory,
        current_generation_size=current_generation_size,
        target_generation_size=target_generation_size,
        desired_generation_size_hint=desired_generation_size_hint,
        cumulative_validated_generated_count=cumulative_validated_generated_count,
        target_problem_count=target_problem_count,
        normalization_mode_hint=normalization_mode_hint,
        selector_feedback=selector_feedback,
    )
    if op_type_allocation is not None:
        trace_inputs["op_type_allocation_recommendation"] = op_type_allocation
    try:
        logger.info("🧠 Strategizing selection...")
        structured_llm = llm.with_structured_output(SelectorPlanSchema, include_raw=True)
        parent = get_current_run_tree()
        selector_run = None
        if parent is not None:
            selector_run = parent.create_child(
                name="orchestrator.plan_generation.llm",
                run_type="chain",
                inputs=trace_inputs,
                tags=["deepagent", "selector", "orchestrator"],
                extra={"metadata": {"stage": "plan_generation", "component": "selector"}},
            )
            selector_run.post()
        try:
            max_retries = 2
            last_exc: Optional[BaseException] = None
            for attempt in range(max_retries + 1):
                try:
                    with tracing_context(enabled=False):
                        response = structured_llm.invoke(messages)
                    if response.get("parsing_error") is not None:
                        raise response["parsing_error"]
                    parsed = response["parsed"]
                    raw_message = response["raw"]
                    strategy = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
                    if selector_run is not None:
                        selector_run.end(outputs=_selector_trace_outputs(strategy, raw_message))
                    return strategy
                except Exception as exc:  # noqa: BLE001
                    if attempt < max_retries and _is_retryable_structured_error(exc):
                        logger.warning(
                            "selector structured-output attempt %d/%d failed with retryable %s; retrying.",
                            attempt + 1, max_retries + 1, type(exc).__name__,
                        )
                        continue
                    last_exc = exc
                    raise
        except Exception as exc:
            if selector_run is not None:
                selector_run.end(
                    outputs={"status": "error"},
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
        finally:
            if selector_run is not None:
                selector_run.patch()
    except Exception as exc:
        logger.warning(f"⚠️ Error: {exc}. Falling back to deterministic pair strategy.")
        return None


def get_last_selection() -> Dict:
    return LAST_SELECTION or {}


def _recommended_normalization_mode(current_generation_size: int, generation_count: int, target_generation_size: int) -> Tuple[str, int]:
    if current_generation_size > target_generation_size:
        return "compress", target_generation_size
    if current_generation_size == target_generation_size:
        return "steady_state", target_generation_size
    return "bootstrap", target_generation_size


def _default_dispatch_fields(item: Dict) -> Dict:
    op_type = item.get("op_type", "mutation")
    slot = int(item.get("slot", 0) or 0)
    mode = item.get("mode") or ("carry" if op_type == "survivor" else "hard")
    difficulty_label = item.get("difficulty_label") or ("survivor" if op_type == "survivor" else ("easy" if mode == "easy" else "hard"))
    target_diff = float(item.get("target_diff", 0.0) or (0.0 if op_type == "survivor" else (6.0 if mode == "easy" else 9.0)))
    target_diff = 0.0 if op_type == "survivor" else _normalize_target_diff(target_diff, difficulty_label, mode)
    # Survivor slots get a fixed literal (axis is unused for carry-over). Non-survivor
    # slots intentionally do NOT receive a generic placeholder when the LLM left the
    # axis empty — the empty string is preserved here so that
    # _validate_and_repair_generation_plan can detect it and trigger a selector replan
    # with explicit feedback. Generic placeholders silently masked the issue and led
    # to meaningless mutations downstream.
    raw_axis = str(item.get("variation_axis", "") or "").strip()
    if op_type == "survivor":
        variation_axis = "preserve elite unchanged"
    else:
        variation_axis = raw_axis  # may be empty; validation step will detect
    difficulty_strategy = item.get("difficulty_strategy") or (
        "unspecified" if op_type == "survivor" else ("easier_stability" if mode == "easy" else "harder_exploratory")
    )
    dispatch_rationale = item.get("dispatch_rationale") or item.get("rationale") or (
        "Preserve the strongest elite candidate unchanged." if op_type == "survivor" else "Dispatch this child to support the orchestrator generation plan."
    )
    execution_group = item.get("execution_group") or (
        "survivor_batch" if op_type == "survivor" else ("crossover_batch" if op_type == "crossover" else "mutation_batch")
    )
    # Plan-grounded research fields; survivor slots must leave both empty.
    raw_seed_focus = str(item.get("seed_focus", "") or "").strip()
    raw_query_hint = str(item.get("query_hint", "") or "").strip()
    if op_type == "survivor":
        seed_focus = ""
        query_hint = ""
    else:
        seed_focus = raw_seed_focus
        # Clamp the query hint to a single line within the schema bound.
        query_hint = " ".join(raw_query_hint.split())[:160]
    return {
        "slot": slot,
        "op_type": op_type,
        "parent_ids": list(item.get("parent_ids", []) or []),
        "mode": mode,
        "difficulty_label": difficulty_label,
        "target_diff": target_diff,
        "variation_axis": variation_axis,
        "difficulty_strategy": difficulty_strategy,
        "dispatch_rationale": dispatch_rationale,
        "execution_group": execution_group,
        "seed_focus": seed_focus,
        "query_hint": query_hint,
    }


def _validate_and_repair_generation_plan(
    plan: Dict,
    current_generation: List[Dict],
    *,
    target_generation_size: int,
    normalization_mode_hint: str,
    desired_generation_size_hint: int,
) -> Tuple[Optional[Dict], List[str], str]:
    if not plan:
        return None, ["selector returned no plan"], "selector"

    strategy_source = "selector"
    repaired = dict(plan)
    current_ids = {problem.get("id") for problem in current_generation if problem.get("id")}
    issues: List[str] = []

    normalization_mode = repaired.get("normalization_mode") or normalization_mode_hint
    if normalization_mode not in {"bootstrap", "compress", "steady_state"}:
        issues.append("normalization_mode must be one of bootstrap/compress/steady_state")
    repaired["normalization_mode"] = normalization_mode
    repaired["target_generation_size"] = target_generation_size

    dispatch_items = [dict(item) for item in (repaired.get("dispatch_items") or [])]
    if not dispatch_items:
        issues.append("dispatch_items must be non-empty")
        return None, issues, strategy_source

    survivor_count = 0
    cleaned_items = []
    seen_slots = set()
    for item in sorted(dispatch_items, key=lambda entry: (int(entry.get("slot", 0) or 0), entry.get("op_type", ""))):
        normalized = _default_dispatch_fields(item)
        slot = normalized["slot"]
        if slot in seen_slots:
            strategy_source = "selector_repaired"
            continue
        seen_slots.add(slot)
        parent_ids = [pid for pid in normalized["parent_ids"] if pid in current_ids]
        if len(parent_ids) != len(normalized["parent_ids"]):
            issues.append(f"dispatch slot {slot} references unknown parent ids")
            continue
        normalized["parent_ids"] = parent_ids
        if normalized["op_type"] == "survivor":
            survivor_count += 1
            if len(parent_ids) != 1:
                issues.append(f"survivor slot {slot} must have exactly one parent")
                continue
        elif normalized["op_type"] == "mutation":
            if len(parent_ids) != 1:
                issues.append(f"mutation slot {slot} must have exactly one parent")
                continue
        elif normalized["op_type"] == "crossover":
            if len(parent_ids) != 2:
                issues.append(f"crossover slot {slot} must have exactly two parents")
                continue
        # Non-survivor slots must have a concrete variation_axis. Empty or stub
        # axes lead to meaningless mutations and frequent axis_missing aborts in
        # the generator (P0 fix from docs/future_improvements.md).
        if normalized["op_type"] != "survivor":
            axis = str(normalized.get("variation_axis", "") or "").strip()
            if len(axis) < 4:
                issues.append(
                    f"slot {slot} ({normalized['op_type']}) has empty or too-short variation_axis "
                    f"'{axis}' — emit a concrete axis grounded in the parent body (>= 4 chars)"
                )
                continue
        cleaned_items.append(normalized)

    if survivor_count > 1:
        issues.append("at most one survivor slot is allowed")
    if survivor_count == 0:
        issues.append("exactly one survivor slot is required")
    if issues:
        return None, issues, strategy_source

    if cleaned_items != dispatch_items:
        strategy_source = "selector_repaired"

    cleaned_items = sorted(cleaned_items, key=lambda item: item["slot"])
    for idx, item in enumerate(cleaned_items):
        if item["slot"] != idx:
            item["slot"] = idx
            strategy_source = "selector_repaired"

    desired_generation_size = repaired.get("desired_generation_size", desired_generation_size_hint)
    try:
        desired_generation_size = int(desired_generation_size)
    except Exception:
        desired_generation_size = len(cleaned_items)
        strategy_source = "selector_repaired"
    if desired_generation_size < 1:
        issues.append("desired_generation_size must be >= 1")
    if desired_generation_size > target_generation_size:
        issues.append("desired_generation_size must be <= target_generation_size")
    if desired_generation_size != len(cleaned_items):
        desired_generation_size = len(cleaned_items)
        strategy_source = "selector_repaired"
    if desired_generation_size < 1 or desired_generation_size > target_generation_size:
        return None, issues or ["invalid desired_generation_size"], strategy_source

    active_pool_ids = []
    for item in cleaned_items:
        for parent_id in item["parent_ids"]:
            if parent_id not in active_pool_ids:
                active_pool_ids.append(parent_id)

    repaired["desired_generation_size"] = desired_generation_size
    repaired["active_pool_ids"] = active_pool_ids
    repaired["dispatch_items"] = cleaned_items
    repaired["plan_rationale"] = repaired.get("plan_rationale") or "Selector plan repaired structurally by orchestrator gate."
    repaired["strategy_source"] = strategy_source
    return repaired, [], strategy_source


def _normalize_answer_key(value) -> str:
    """Canonical comparator for survivor answers.

    Whitespace-collapsed, lowercased, stripped. Empty input → empty string
    (treated as no-op by the caller).
    """
    return " ".join(str(value or "").strip().lower().split())


def _enforce_fixed_point_depth(
    plan: Dict,
    current_generation: List[Dict],
) -> Tuple[Dict, Optional[str]]:
    """Block fixed_point_descendant problems from becoming survivor (P3.4).

    Fixed-point seeds (e.g. AC-1 with answer mathematically forced to -1)
    propagate their constant answer to all descendants. Allowing such a
    descendant to carry forward as survivor recreates the same chain. We
    permit depth-1 descendants to exist (so the seed's family is explored)
    but never let them become the survivor pick.
    """
    if not plan:
        return plan, None
    dispatch_items = plan.get("dispatch_items") or []
    survivor_item = next((item for item in dispatch_items if item.get("op_type") == "survivor"), None)
    if survivor_item is None or not survivor_item.get("parent_ids"):
        return plan, None
    survivor_pid = survivor_item["parent_ids"][0]
    prob_by_id = {p.get("id"): p for p in current_generation if p.get("id")}
    picked = prob_by_id.get(survivor_pid)
    if not picked:
        return plan, None
    is_fp = bool(picked.get("fixed_point") or picked.get("fixed_point_descendant"))
    if not is_fp:
        return plan, None
    # Find replacement with no fixed_point lineage
    alternatives = [
        p for p in current_generation
        if p.get("id") and p.get("id") != survivor_pid
        and not p.get("fixed_point") and not p.get("fixed_point_descendant")
    ]
    if not alternatives:
        return plan, (
            f"Survivor pick '{survivor_pid}' has fixed_point lineage but no clean "
            f"alternative elite is available. Generation may collapse."
        )
    # Pick highest-quality clean alternative
    def _qscore(p):
        q = p.get("quality_assessment") or {}
        return float(q.get("score", 0.0) or 0.0)
    alternatives.sort(key=lambda p: (-_qscore(p),))
    replacement = alternatives[0]
    new_id = replacement.get("id")
    survivor_item["parent_ids"] = [new_id]
    survivor_item["dispatch_rationale"] = (
        (survivor_item.get("dispatch_rationale", "") or "")
        + (" | " if survivor_item.get("dispatch_rationale") else "")
        + f"[fixed_point_guard] swapped to {new_id}: previous pick '{survivor_pid}' has fixed_point lineage."
    )
    plan["dispatch_items"] = dispatch_items
    plan["fixed_point_guard_swap"] = {
        "from_id": survivor_pid,
        "to_id": new_id,
        "reason": "fixed_point_lineage",
    }
    record_deterministic_span(
        "selector.fixed_point_guard",
        inputs={
            "survivor_pick_id": survivor_pid,
            "survivor_pick_fixed_point": bool(picked.get("fixed_point")),
            "survivor_pick_fixed_point_descendant": bool(picked.get("fixed_point_descendant")),
            "candidate_pool_size": len(current_generation),
        },
        outputs={
            "swapped": True,
            "to_id": new_id,
            "reason": "fixed_point_lineage — blocked to prevent constant-answer chain",
        },
        tags=["deepagent", "selector", "fixed_point_guard"],
    )
    return plan, None


def _enforce_quality_parent_guard(
    plan: Dict,
    current_generation: List[Dict],
) -> Tuple[Dict, Optional[str]]:
    """Swap quality_low parents to higher-quality alternatives when available.

    Problems saved with quality_assessment.passed=False carry quality_low=True.
    Using them as parents tends to produce near-copies (similarity >= 0.97) that
    the near-copy gate blocks, wasting a generation slot. This guard swaps them
    out pre-emptively when a clean alternative exists.
    """
    if not plan:
        return plan, None
    dispatch_items = plan.get("dispatch_items") or []
    prob_by_id = {p.get("id"): p for p in current_generation if p.get("id")}
    # Clean candidates: not quality_low, not fixed_point lineage
    clean_pool = [
        p for p in current_generation
        if not p.get("quality_low") and not p.get("fixed_point") and not p.get("fixed_point_descendant")
    ]
    warnings: List[str] = []
    swapped_any = False
    for item in dispatch_items:
        if item.get("op_type") == "survivor":
            continue
        parent_ids = list(item.get("parent_ids") or [])
        low_pids = [pid for pid in parent_ids if prob_by_id.get(pid, {}).get("quality_low")]
        if not low_pids:
            continue
        for low_pid in low_pids:
            in_use = set(parent_ids) - {low_pid}
            alts = [p for p in clean_pool if p.get("id") not in in_use and p.get("id") != low_pid]
            if alts:
                replacement = alts[0]
                new_pid = replacement.get("id")
                parent_ids = [new_pid if pid == low_pid else pid for pid in parent_ids]
                item["parent_ids"] = parent_ids
                item["dispatch_rationale"] = (
                    (item.get("dispatch_rationale") or "")
                    + f" | [quality_guard] swapped parent {low_pid} → {new_pid} (quality_low)"
                )
                swapped_any = True
                record_deterministic_span(
                    "selector.quality_parent_guard",
                    inputs={"slot": item.get("slot"), "low_pid": low_pid, "pool_size": len(clean_pool)},
                    outputs={"swapped": True, "to_id": new_pid},
                    tags=["deepagent", "selector", "quality_guard"],
                )
            else:
                warnings.append(f"slot {item.get('slot')}: quality_low parent {low_pid} kept — no clean alternative")
    if swapped_any:
        plan["dispatch_items"] = dispatch_items
    warning = "; ".join(warnings) if warnings else None
    return plan, warning


def _enforce_survivor_answer_guard(
    plan: Dict,
    current_generation: List[Dict],
    recent_survivor_answers: List[str],
    *,
    two_gen_ban: bool = True,
) -> Tuple[Dict, Optional[str]]:
    """Swap the survivor pick when its answer would extend a collapse chain.

    Policy (P3.2): if the picked survivor's answer equals the most recent saved
    survivor answer, look for a replacement elite from current_generation with a
    different answer. If no replacement exists, leave the plan unchanged and
    surface a warning. With `two_gen_ban=True`, a 2-in-a-row repeat is blocked
    (not just 3-in-a-row), because the same-answer chain is the observed failure
    mode — waiting until 3 is already a collapse in progress.

    Returns (possibly-mutated plan, warning_or_none). Never raises.
    """
    if not plan or not recent_survivor_answers:
        return plan, None
    dispatch_items = plan.get("dispatch_items") or []
    survivor_item = next((item for item in dispatch_items if item.get("op_type") == "survivor"), None)
    if survivor_item is None:
        return plan, None
    parent_ids = survivor_item.get("parent_ids") or []
    if not parent_ids:
        return plan, None
    survivor_parent_id = parent_ids[0]
    prob_by_id = {p.get("id"): p for p in current_generation if p.get("id")}
    picked = prob_by_id.get(survivor_parent_id)
    if not picked:
        return plan, None
    picked_ans = _normalize_answer_key(picked.get("answer"))
    if not picked_ans:
        return plan, None
    recent_keys = [_normalize_answer_key(a) for a in recent_survivor_answers if a]
    if not recent_keys:
        return plan, None
    last_answer = recent_keys[-1]
    hard_collision = picked_ans == last_answer
    if not hard_collision and two_gen_ban and len(recent_keys) >= 2:
        hard_collision = picked_ans == recent_keys[-2] == last_answer
    if not hard_collision:
        return plan, None

    # Look for an alternative elite with a different answer. Preference order:
    # (1) non-fixed_point AND non-fixed_point_descendant, (2) highest-quality
    # if quality_assessment exists, (3) lowest difficulty (easier elite is
    # safer as survivor).
    def _score(problem: Dict) -> Tuple[int, float, float]:
        fx_seed = 1 if problem.get("fixed_point") else 0
        fx_desc = 1 if problem.get("fixed_point_descendant") else 0
        # Combine: any fixed_point lineage is heavily penalized
        fx = fx_seed + fx_desc
        q = problem.get("quality_assessment") or {}
        qs = float(q.get("score", 0.0) or 0.0)
        try:
            diff = float(problem.get("difficulty", 0.0) or 0.0)
        except (ValueError, TypeError):
            diff = 0.0
        return (fx, -qs, diff)

    alternatives = [
        p for p in current_generation
        if p.get("id") and p.get("id") != survivor_parent_id
        and _normalize_answer_key(p.get("answer")) not in {picked_ans}
        and _normalize_answer_key(p.get("answer"))  # non-empty
    ]
    if not alternatives:
        warning = (
            f"Survivor answer '{picked.get('answer','')}' collides with recent chain "
            f"{recent_survivor_answers[-2:]} but no alternative elite with a different "
            f"answer is available in the current generation."
        )
        record_deterministic_span(
            "selector.survivor_answer_guard",
            inputs={
                "survivor_pick_id": survivor_parent_id,
                "survivor_pick_answer": str(picked.get("answer", ""))[:60],
                "recent_survivor_answers": list(recent_survivor_answers[-3:]),
                "alternatives_count": 0,
            },
            outputs={
                "swapped": False,
                "reason": "no_alternative_available",
                "warning": warning,
            },
            tags=["deepagent", "selector", "survivor_guard", "warning"],
        )
        return plan, warning
    alternatives.sort(key=_score)
    replacement = alternatives[0]
    new_survivor_id = replacement.get("id")
    survivor_item["parent_ids"] = [new_survivor_id]
    survivor_item["dispatch_rationale"] = (
        (survivor_item.get("dispatch_rationale", "") or "")
        + (" | " if survivor_item.get("dispatch_rationale") else "")
        + f"[survivor_guard] swapped to {new_survivor_id}: previous pick answer '{picked.get('answer','')}' matched recent survivor chain."
    )
    plan["dispatch_items"] = dispatch_items
    plan["survivor_guard_swap"] = {
        "from_id": survivor_parent_id,
        "to_id": new_survivor_id,
        "reason": "answer_chain_collision",
        "blocked_answer": picked.get("answer", ""),
        "recent_survivor_answers": list(recent_survivor_answers[-3:]),
    }
    record_deterministic_span(
        "selector.survivor_answer_guard",
        inputs={
            "survivor_pick_id": survivor_parent_id,
            "survivor_pick_answer": str(picked.get("answer", ""))[:60],
            "recent_survivor_answers": list(recent_survivor_answers[-3:]),
            "alternatives_count": len(alternatives),
        },
        outputs={
            "swapped": True,
            "to_id": new_survivor_id,
            "to_answer": str(replacement.get("answer", ""))[:60],
            "reason": "answer_chain_collision — swapped to prevent same-answer survivor chain",
        },
        tags=["deepagent", "selector", "survivor_guard"],
    )
    return plan, None


def plan_generation(
    current_generation: List[Dict],
    parameters: Dict = None,
    invariant_bundles: Optional[Dict[str, Dict]] = None,
    run_working_memory: Optional[Dict] = None,
    plan_outcome_cards: Optional[List[Dict]] = None,
    recent_survivor_answers: Optional[List[str]] = None,
) -> Dict:
    if parameters is None:
        parameters = {}
    mutation_only = bool(parameters.get("mutation_only"))
    target_generation_size = _canonical_target_generation_size(parameters)
    generation_count = int(parameters.get("generation_count", parameters.get("gen_count", 0)) or 0)
    current_generation_size = len(current_generation)
    normalization_mode_hint, desired_generation_size_hint = _recommended_normalization_mode(
        current_generation_size,
        generation_count,
        target_generation_size,
    )
    target_problem_count = parameters.get("target_problem_count")
    cumulative_validated_generated_count = int(parameters.get("cumulative_validated_generated_count", 0) or 0)

    # Compute the adaptive mutation/crossover allocation hint before invoking
    # the selector LLM. Non-survivor slots = desired_generation_size_hint - 1
    # (exactly one survivor is always reserved).
    non_survivor_slots = max(0, int(desired_generation_size_hint) - 1)
    from deepagent.memory_bank import recommend_op_type_allocation
    op_type_allocation = recommend_op_type_allocation(
        list(plan_outcome_cards or []),
        seed_count=current_generation_size,
        non_survivor_slots=non_survivor_slots,
    )
    if mutation_only:
        # Force mutation-only regardless of history when the user explicitly
        # requested it via --mutation-only.
        op_type_allocation = {
            "mutation": non_survivor_slots,
            "crossover": 0,
            "rationale": "mutation_only flag set by caller; ignoring historical op_type stats.",
            "confidence": "hard_constraint",
            "stats_available": False,
            "observed_attempts": op_type_allocation.get("observed_attempts", {"mutation": 0, "crossover": 0}),
        }

    selector_feedback = ""
    final_plan = None
    strategy_source = "selector"
    for attempt in range(2):
        plan = get_selection_strategy(
            current_generation,
            invariant_bundles=invariant_bundles,
            run_working_memory=run_working_memory,
            current_generation_size=current_generation_size,
            target_generation_size=target_generation_size,
            desired_generation_size_hint=desired_generation_size_hint,
            cumulative_validated_generated_count=cumulative_validated_generated_count,
            target_problem_count=target_problem_count,
            normalization_mode_hint=normalization_mode_hint,
            selector_feedback=selector_feedback,
            op_type_allocation=op_type_allocation,
            recent_survivor_answers=list(recent_survivor_answers or []),
        )
        validated_plan, issues, strategy_source = _validate_and_repair_generation_plan(
            plan,
            current_generation,
            target_generation_size=target_generation_size,
            normalization_mode_hint=normalization_mode_hint,
            desired_generation_size_hint=desired_generation_size_hint,
        )
        if validated_plan is not None:
            final_plan = validated_plan
            if attempt == 1 and strategy_source == "selector":
                strategy_source = "selector_replanned"
                final_plan["strategy_source"] = strategy_source
            break
        selector_feedback = "\n".join(f"- {issue}" for issue in issues) or "- selector returned an unusable plan"
        if attempt == 0:
            logger.warning("Selector generation plan invalid; requesting one replan with validation feedback.")

    if final_plan is None:
        raise ValueError(f"Selector failed to produce a valid generation plan after one replan.\n{selector_feedback}")

    # Swap quality_low parents to clean alternatives before other guards fire.
    # Runs first so that fixed_point_depth and survivor_answer guards operate
    # on a plan that already has the best available parents.
    final_plan, quality_warning = _enforce_quality_parent_guard(final_plan, current_generation)
    if quality_warning:
        logger.warning(f"quality_parent_guard: {quality_warning}")

    # P3.4 — block fixed_point lineage from carrying forward as survivor. Runs
    # BEFORE the answer-chain guard so that the answer-chain guard sees a clean
    # survivor pick.
    final_plan, fp_warning = _enforce_fixed_point_depth(final_plan, current_generation)
    if fp_warning:
        logger.warning(f"fixed_point_guard: {fp_warning}")

    # P3.2 — defend against survivor-answer collapse chains. Runs AFTER the
    # selector has produced a structurally valid plan; only mutates the survivor
    # slot's parent_ids when a collision is detected.
    final_plan, guard_warning = _enforce_survivor_answer_guard(
        final_plan,
        current_generation,
        list(recent_survivor_answers or []),
    )
    if guard_warning:
        logger.warning(f"survivor_answer_guard: {guard_warning}")

    dispatch_items = [dict(item) for item in (final_plan.get("dispatch_items") or [])]
    prob_map = {problem.get("id"): problem for problem in current_generation if problem.get("id")}
    work_items = []
    for item in dispatch_items:
        parent_ids = item.get("parent_ids", []) or []
        parents = [prob_map[parent_id] for parent_id in parent_ids if parent_id in prob_map]
        work_items.append(
            {
                **item,
                "pair_id": item.get("pair_id") or f"{item.get('execution_group', 'batch')}_slot_{item.get('slot', 0)}",
                "parents": [_slim_parent(parent) for parent in parents],
                "requires_research": item.get("op_type") != "survivor",
            }
        )

    pair_plans = []

    final_plan["strategy_source"] = strategy_source
    # Record recommended vs chosen allocation on the strategy so downstream
    # artifacts and selector trace outputs expose selector compliance.
    chosen_mutation = sum(1 for item in dispatch_items if item.get("op_type") == "mutation")
    chosen_crossover = sum(1 for item in dispatch_items if item.get("op_type") == "crossover")
    recommended_mutation = int(op_type_allocation.get("mutation", 0)) if op_type_allocation else 0
    recommended_crossover = int(op_type_allocation.get("crossover", 0)) if op_type_allocation else 0
    recommendation_followed = (
        chosen_mutation == recommended_mutation and chosen_crossover == recommended_crossover
    )
    final_plan["op_type_allocation"] = {
        "recommended": {
            "mutation": recommended_mutation,
            "crossover": recommended_crossover,
            "confidence": op_type_allocation.get("confidence", "default") if op_type_allocation else "default",
            "rationale": op_type_allocation.get("rationale", "") if op_type_allocation else "",
            "observed_attempts": (op_type_allocation.get("observed_attempts", {}) if op_type_allocation else {}),
        },
        "chosen": {"mutation": chosen_mutation, "crossover": chosen_crossover},
        "recommendation_followed": recommendation_followed,
    }
    global LAST_SELECTION
    LAST_SELECTION = final_plan
    return {
        "strategy": final_plan,
        "pair_plans": pair_plans,
        "work_items": work_items,
        "desired_generation_size": int(final_plan.get("desired_generation_size", len(dispatch_items))),
    }
