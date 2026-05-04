"""Pure helper functions extracted from graph_full.py."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from deepagent.quality import assess_problem_quality
from deepagent.state_full import AgentState
from deepagent.graph.runtime import (
    LINEAGE_DEPTH_EASY_THRESHOLD,
    CROSSOVER_DEPTH_EASY_THRESHOLD,
    _coerce_difficulty_value,
    _desired_generation_size,
)

logger = logging.getLogger(__name__)


def _fallback_history(problem: Dict) -> List[str]:
    history = list(problem.get("fallback_history", []) or [])
    if problem.get("fallback_source_id"):
        history.append(str(problem.get("fallback_source_id")))
    return history


def _ensure_unique_problem_id(problem: Dict, state: AgentState, existing_candidates: List[Dict] = None, failed_problems: List[Dict] = None) -> Dict:
    existing_ids = {
        p.get("id")
        for p in (state.get("current_generation", []) or [])
        if p.get("id")
    }
    existing_ids.update(
        candidate.get("id")
        for candidate in (existing_candidates or [])
        if candidate.get("id")
    )
    existing_ids.update(
        problem_item.get("id")
        for problem_item in (failed_problems or [])
        if problem_item.get("id")
    )
    existing_ids.discard(problem.get("id"))

    base_id = problem.get("id", "")
    if not base_id:
        return problem
    candidate_id = base_id
    suffix = 2
    while candidate_id in existing_ids:
        candidate_id = f"{base_id}_r{suffix}"
        suffix += 1
    problem["id"] = candidate_id
    return problem


def _reason_signature(reason: str) -> str:
    text = (reason or "").strip().lower()
    if "quality gate failed" in text:
        return "quality_gate"
    if "novelty collapse" in text:
        return "novelty_collapse"
    if "near-copy" in text or "near copy" in text:
        return "near_copy"
    if "too_easy_derivative" in text or "too easy" in text:
        return "too_easy"
    if (
        "target quantity" in text
        or "different mathematical task" in text
        or "sum of traces" in text
        or "trace summation" in text
        or "proxy target" in text
        or "helper statistic" in text
    ):
        return "target_quantity"
    if (
        "different mathematical problem" in text
        or "underlying symmetric system" in text
        or "natural numbers as the domain" in text
        or "real numbers" in text
        or "domain constraints" in text
    ):
        return "statement_domain"
    if "natural numbers" in text or "real numbers" in text or "domain" in text:
        return "domain_drift"
    if "tie-breaking rule" in text or "tie breaking rule" in text:
        return "missing_tie_break"
    if "no problem statement provided" in text or "missing statement/answer/solution" in text:
        return "missing_required_fields"
    if "validation error" in text:
        return "validation_error"
    return text[:120]


def _failure_type_from_reason(reason: str, *, fallback: str = "regenerability_failure") -> str:
    text = (reason or "").strip().lower()
    if "missing or placeholder fields" in text or "missing statement/answer/solution" in text or "no problem statement provided" in text:
        return "missing_required_fields"
    if "statement/code inconsistency" in text:
        return "statement_code_inconsistency"
    if "statement/solution inconsistency" in text:
        return "statement_solution_inconsistency"
    if "does not appear in the statement" in text or "hardcodes derived quantity" in text:
        return "hidden_constant_failure"
    if "residue" in text and "expected" in text:
        return "residue_mismatch_failure"
    if "no cyclically divisible triple" in text or "no natural-number solution" in text or "no solution under stated constraints" in text:
        return "no_solution_under_stated_constraints"
    if "solvability failed" in text or "constraint system is inconsistent" in text:
        return "solvability_failure"
    if "novelty collapse failed" in text or "novelty collapse" in text:
        return "novelty_collapse_failure"
    if (
        "target quantity" in text
        or "different mathematical task" in text
        or "sum of traces" in text
        or "trace summation" in text
        or "proxy target" in text
        or "helper statistic" in text
    ):
        return "target_quantity_failure"
    if (
        "different mathematical problem" in text
        or "underlying symmetric system" in text
        or "natural numbers as the domain" in text
        or "real numbers" in text
        or "domain constraints" in text
    ):
        return "statement_domain_failure"
    if "hardcode" in text or "numeric constant directly" in text or "sham" in text:
        return "sham_code_failure"
    if "exploratory python" in text or "exploration" in text:
        return "exploration_failure"
    if "code gate" in text or "correctness gate failed" in text or "answer materialization failed" in text:
        return "code_gate_failure"
    if "exact relation" in text or "subset-style" in text or "iff" in text or "exactly" in text:
        return "relation_guard_failure"
    if "invariant" in text or "forbidden rewrite" in text or "constraint guard" in text:
        return "invariant_failure"
    if "regenerability failed" in text:
        return "regenerability_failure"
    if "quality gate failed" in text or "near-copy" in text or "near copy" in text or "too easy" in text:
        return "quality_failure"
    return fallback


def _slot_failure_registry(state: AgentState) -> Dict:
    return dict(state.get("slot_failure_registry", {}) or state.get("slot_retry_registry", {}) or state.get("retry_registry", {}) or {})


def _record_slot_failure(
    registry: Dict,
    slot: int,
    *,
    failure_type: str,
    failure_stage: str,
    failure_reason: str,
    repair_strategy: str = "",
) -> Dict:
    slot_key = str(slot)
    entry = dict(registry.get(slot_key, {}) or {})
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    entry["failure_types"] = list((entry.get("failure_types", []) or []) + [failure_type])[-6:]
    entry["failure_stages"] = list((entry.get("failure_stages", []) or []) + [failure_stage])[-6:]
    entry["reason_signatures"] = list((entry.get("reason_signatures", []) or []) + [_reason_signature(failure_reason)])[-6:]
    entry["last_reason"] = failure_reason
    entry["last_failure_type"] = failure_type
    entry["last_failure_stage"] = failure_stage
    entry["last_repair_strategy"] = repair_strategy
    registry[slot_key] = entry
    return registry


def _record_slot_decision(registry: Dict, slot: int, decision: str) -> Dict:
    """Push the latest regen_planner decision onto the slot's history.

    The history is read by `_build_slot_failure_summary` and surfaced to the
    next regen_planner cycle as `prior_decisions`, letting it detect and avoid
    repeating ineffective routing.
    """
    if not decision:
        return registry
    slot_key = str(slot)
    entry = dict(registry.get(slot_key, {}) or {})
    entry["decisions"] = list((entry.get("decisions", []) or []) + [str(decision)])[-6:]
    registry[slot_key] = entry
    return registry


def _lineage_metrics_for_problem(problem: Dict) -> Dict:
    metrics = dict(problem.get("lineage_metrics", {}) or {})
    metrics.setdefault("lineage_depth", 0)
    metrics.setdefault("crossover_depth", 0)
    metrics.setdefault("mutation_depth", 0)
    metrics.setdefault("recent_retry_reasons", [])
    metrics.setdefault("recent_parent_ids", problem.get("parent_ids", []) or [])
    return metrics


def _lineage_index(problems: List[Dict]) -> Dict[str, Dict]:
    index = {}
    for problem in problems or []:
        problem_id = problem.get("id")
        if problem_id:
            index[problem_id] = _lineage_metrics_for_problem(problem)
    return index


def _derive_child_lineage_metrics(parents: List[Dict], op_type: str, retry_reasons: List[str] = None) -> Dict:
    parent_metrics = [_lineage_metrics_for_problem(parent) for parent in parents or []]
    lineage_depth = (max((metrics.get("lineage_depth", 0) for metrics in parent_metrics), default=0) + 1) if parents else 0
    crossover_depth = max((metrics.get("crossover_depth", 0) for metrics in parent_metrics), default=0)
    mutation_depth = max((metrics.get("mutation_depth", 0) for metrics in parent_metrics), default=0)
    if op_type == "crossover":
        crossover_depth += 1
    elif op_type == "mutation":
        mutation_depth += 1
    return {
        "lineage_depth": lineage_depth,
        "crossover_depth": crossover_depth,
        "mutation_depth": mutation_depth,
        "recent_retry_reasons": list(retry_reasons or [])[-4:],
        "recent_parent_ids": [parent.get("id") for parent in parents or [] if parent.get("id")],
    }


def _all_invariant_flags_true(audit: Dict) -> bool:
    return bool(audit) and all(
        [
            audit.get("definition_preserved", False),
            audit.get("domain_preserved", False),
            audit.get("relation_preserved", False),
            audit.get("target_quantity_preserved", False),
        ]
    )


def _quality_issues_for_pair(candidate: Dict, parents: List[Dict], target_diff: float, invoke_config: Dict = None) -> Tuple[List[str], Dict]:
    try:
        _, _, assessment = assess_problem_quality(candidate, parents, target_diff, invoke_config=invoke_config)
        return list(assessment.get("issues") or []), assessment
    except Exception as exc:
        logger.warning(f"Pair quality precheck failed: {exc}")
        return ["other"], {
            "passed": False,
            "issues": ["other"],
            "reason": f"Pair quality precheck failed: {exc}",
            "novelty_score": 1,
            "difficulty_alignment_score": 1,
        }


def _make_failed_problem(item: Dict, reason: str, extra: Dict = None) -> Dict:
    slot = item.get("_slot", item.get("slot", 0))
    failed = {
        "id": item.get("id", ""),
        "_slot": slot,
        "slot": slot,
        "op_type": item.get("op_type", ""),
        "type": item.get("type", item.get("op_type", "")),
        "pair_id": item.get("pair_id"),
        "parent_ids": list(item.get("parent_ids", []) or []),
        "validation_feedback": reason,
        "constraint_guard_feedback": item.get("constraint_guard_feedback", ""),
        "difficulty_label": item.get("difficulty_label", ""),
        "difficulty": item.get("target_diff", item.get("difficulty", 0)),
        "mode": item.get("mode", ""),
        "shared_invariant": item.get("shared_invariant", ""),
        "bridge_axis": item.get("bridge_axis", ""),
        "variation_axis": item.get("variation_axis", ""),
        "mutation_parent_id": item.get("mutation_parent_id"),
        "difficulty_strategy": item.get("difficulty_strategy", ""),
        "dispatch_rationale": item.get("dispatch_rationale", ""),
        "failure_signature": _reason_signature(reason),
        "failure_type": item.get("worker_failure_type", "") or _failure_type_from_reason(reason),
        "failure_stage": item.get("worker_failure_stage", "") or "graph",
        "failure_reason": item.get("worker_failure_reason", "") or reason,
        "repair_strategy": item.get("repair_strategy", ""),
    }
    if extra:
        failed.update(extra)
    return failed


def _score_elite_backfill_candidate(problem: Dict, failed_problem: Dict, used_ids: set[str], slot_registry: Dict) -> Tuple[float, List[str]]:
    problem_id = problem.get("id")
    if not problem_id or problem_id in used_ids:
        return float("-inf"), ["already_used"]
    if problem.get("type") == "fallback_survivor":
        return float("-inf"), ["prior_fallback"]
    if _fallback_history(problem):
        return float("-inf"), ["fallback_history"]

    score = 0.0
    reasons = []
    failed_parent_ids = set(failed_problem.get("parent_ids", []) or [])
    failed_mutation_parent = failed_problem.get("mutation_parent_id")
    if problem_id == failed_mutation_parent:
        score += 5.0
        reasons.append("mutation_parent")
    elif problem_id in failed_parent_ids:
        score += 4.0
        reasons.append("failed_parent")

    metrics = _lineage_metrics_for_problem(problem)
    score -= 0.35 * float(metrics.get("lineage_depth", 0) or 0)
    score -= 0.25 * float(metrics.get("crossover_depth", 0) or 0)
    score -= 0.2 * float(metrics.get("mutation_depth", 0) or 0)

    retry_signatures = list(slot_registry.get("reason_signatures", []) or [])
    if any(sig in {"near_copy", "too_easy", "quality_gate"} for sig in retry_signatures):
        score += 0.75
        reasons.append("stability_preferred")

    quality = dict(problem.get("quality_assessment", {}) or {})
    quality_issues = set(quality.get("issues", []) or [])
    if "near_copy" in quality_issues:
        score -= 2.0
        reasons.append("near_copy_penalty")
    if "novelty_low" in quality_issues:
        score -= 0.8
        reasons.append("novelty_penalty")

    if failed_problem.get("op_type") == "crossover" and problem.get("type") == "survivor":
        score += 0.5
        reasons.append("survivor_safe")

    score += max(0.0, min(2.0, _coerce_difficulty_value(problem.get("difficulty", 0), default=0.0) / 10.0))
    return score, reasons


def _select_elite_backfill(state: AgentState, approved: List[Dict], failed: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    params = dict(state.get("parameters", {}) or {})
    required_population = _desired_generation_size(state, params)
    missing_slots = required_population - len(approved)
    if missing_slots <= 0:
        return approved, failed, []
    if not failed:
        return approved, failed, []

    current_generation = list(state.get("current_generation", []) or [])
    used_ids = {
        (item.get("problem") or {}).get("id")
        for item in approved
        if (item.get("problem") or {}).get("id")
    }
    slot_registry_map = _slot_failure_registry(state)
    fallback_events = []
    remaining_failed = list(failed)

    for failed_problem in list(failed):
        slot = int(failed_problem.get("_slot", failed_problem.get("slot", -1)) or -1)
        if slot < 0:
            continue
        candidates = []
        for problem in current_generation:
            score, reasons = _score_elite_backfill_candidate(
                problem,
                failed_problem,
                used_ids=used_ids,
                slot_registry=dict(slot_registry_map.get(str(slot), {}) or {}),
            )
            if score != float("-inf"):
                candidates.append((score, reasons, problem))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item[0], item[2].get("id", "")))
        top_score, reasons, selected = candidates[0]
        fallback_problem = dict(selected)
        fallback_problem["_slot"] = slot
        fallback_problem["type"] = "fallback_survivor"
        fallback_problem["op_type"] = "fallback_survivor"
        fallback_problem["fallback_source_id"] = selected.get("id")
        fallback_problem["fallback_reason"] = failed_problem.get("validation_feedback") or failed_problem.get("constraint_guard_feedback") or "elite_backfill"
        fallback_problem["fallback_score"] = round(top_score, 3)
        fallback_problem["fallback_from_generation"] = state.get("generation_count", 0)
        fallback_problem["fallback_target_slot"] = slot
        fallback_problem["fallback_selection_reasons"] = reasons
        fallback_problem["fallback_history"] = list(dict(selected).get("fallback_history", []) or []) + [selected.get("id")]
        fallback_problem["lineage_metrics"] = _lineage_metrics_for_problem(selected)
        approved.append({"slot": slot, "problem": fallback_problem})
        used_ids.add(selected.get("id"))
        remaining_failed = [
            item for item in remaining_failed
            if int(item.get("_slot", item.get("slot", -1)) or -1) != slot
        ]
        fallback_events.append(
            {
                "slot": slot,
                "source_id": selected.get("id"),
                "score": round(top_score, 3),
                "reasons": reasons,
            }
        )

    approved = sorted(approved, key=lambda item: item.get("slot", 0))
    return approved, remaining_failed, fallback_events


def _prune_failed_for_approved_slots(approved_by_slot: Dict[int, Dict], failed: List[Dict]) -> List[Dict]:
    approved_slots = {int(slot) for slot in approved_by_slot.keys()}
    pruned = []
    for item in failed or []:
        slot = int(item.get("_slot", item.get("slot", -1)) or -1)
        if slot in approved_slots:
            continue
        pruned.append(item)
    return pruned


def _clear_nonblocking_failed_when_population_full(
    approved_by_slot: Dict[int, Dict],
    failed: List[Dict],
    required_population: int,
) -> List[Dict]:
    if len(approved_by_slot) >= required_population:
        return []
    return failed


def _pair_health_score(
    parents: List[Dict],
    crossover_candidate: Dict = None,
    crossover_error: str = "",
    retry_signatures: List[str] = None,
    quality_issues: List[str] = None,
) -> float:
    score = 0.6
    retry_signatures = retry_signatures or []
    quality_issues = quality_issues or []
    if crossover_error:
        score -= 0.35
    if crossover_candidate and not _all_invariant_flags_true(crossover_candidate.get("invariant_audit", {}) or {}):
        score -= 0.2
    max_lineage = max((_lineage_metrics_for_problem(parent).get("lineage_depth", 0) for parent in parents or []), default=0)
    max_crossover = max((_lineage_metrics_for_problem(parent).get("crossover_depth", 0) for parent in parents or []), default=0)
    if max_lineage >= LINEAGE_DEPTH_EASY_THRESHOLD:
        score -= 0.15
    if max_crossover >= CROSSOVER_DEPTH_EASY_THRESHOLD:
        score -= 0.15
    if any(issue in {"novelty_low", "too_easy_derivative"} for issue in quality_issues):
        score += 0.1
    if any(signature in {"quality_gate", "near_copy", "too_easy"} for signature in retry_signatures):
        score += 0.1
    if any(signature in {"validation_error", "domain_drift", "target_quantity", "missing_required_fields"} for signature in retry_signatures):
        score -= 0.1
    return max(0.0, min(1.0, score))


def _determine_mutation_policy(pair_plan: Dict, state: AgentState, crossover_candidate: Dict = None, crossover_error: str = "") -> Dict:
    mutation_parent = pair_plan.get("mutation_parent") or {}
    parent_metrics = _lineage_metrics_for_problem(mutation_parent)
    mutation_slot = pair_plan.get("mutation_slot", 0)
    slot_registry = dict(_slot_failure_registry(state).get(str(mutation_slot), {}) or {})
    retry_signatures = list(slot_registry.get("reason_signatures", []) or [])
    quality_issues = []
    quality_assessment = {}
    if crossover_candidate and not crossover_error:
        quality_issues, quality_assessment = _quality_issues_for_pair(
            crossover_candidate,
            pair_plan.get("parents", []),
            _coerce_difficulty_value(crossover_candidate.get("difficulty", pair_plan.get("crossover_target_diff", 9.0)), default=9.0),
            invoke_config={
                "run_name": f"deepagent.quality.{pair_plan.get('pair_id', 'pair')}",
                "tags": ["deepagent", "quality", pair_plan.get("pair_id", "pair")],
                "metadata": {"pair_id": pair_plan.get("pair_id"), "slot": pair_plan.get("crossover_slot")},
            },
        )
        crossover_candidate["pair_quality_assessment"] = quality_assessment

    easier_due_to_failure = bool(crossover_error) or (
        crossover_candidate is not None and not _all_invariant_flags_true(crossover_candidate.get("invariant_audit", {}) or {})
    )
    easier_due_to_complexity = (
        parent_metrics.get("lineage_depth", 0) >= LINEAGE_DEPTH_EASY_THRESHOLD
        or parent_metrics.get("crossover_depth", 0) >= CROSSOVER_DEPTH_EASY_THRESHOLD
    )
    harder_due_to_novelty = any(issue in {"novelty_low", "too_easy_derivative"} for issue in quality_issues) or any(
        signature in {"quality_gate", "near_copy", "too_easy"} for signature in retry_signatures
    )
    easier_due_to_retries = any(
        signature in {"validation_error", "domain_drift", "target_quantity", "missing_required_fields"} for signature in retry_signatures
    )

    direction = "harder"
    rationale = "Default exploratory mutation after successful crossover."
    if easier_due_to_failure:
        direction = "easier"
        rationale = "Crossover failed or showed invariant drift risk, so dispatching an easier rescue mutation."
    elif easier_due_to_complexity:
        direction = "easier"
        rationale = "Parent lineage/crossover depth is high, so lowering mutation difficulty for stability."
    elif easier_due_to_retries:
        direction = "easier"
        rationale = "Recent mutation retries failed on correctness or constraint checks, so lowering difficulty."
    elif harder_due_to_novelty:
        direction = "harder"
        rationale = "Crossover completed but looked too derivative, so increasing mutation difficulty to recover novelty."

    parent_diff = _coerce_difficulty_value(mutation_parent.get("difficulty", 7.0), default=7.0)
    if direction == "easier":
        target_diff = min(parent_diff, 6.0)
        difficulty_label = "easy"
        mode = "easy"
    else:
        target_diff = max(parent_diff, 8.5)
        difficulty_label = "hard"
        mode = "hard"

    pair_health = _pair_health_score(
        pair_plan.get("parents", []),
        crossover_candidate=crossover_candidate,
        crossover_error=crossover_error,
        retry_signatures=retry_signatures,
        quality_issues=quality_issues,
    )
    return {
        "pair_id": pair_plan.get("pair_id"),
        "mutation_parent_id": mutation_parent.get("id"),
        "mutation_slot": mutation_slot,
        "mutation_direction": direction,
        "difficulty_strategy": "easier_rescue" if direction == "easier" and easier_due_to_failure else (
            "easier_stability" if direction == "easier" else (
                "harder_novelty_recovery" if harder_due_to_novelty else "harder_exploratory"
            )
        ),
        "mode": mode,
        "difficulty_label": difficulty_label,
        "target_diff": target_diff,
        "rationale": rationale,
        "quality_issues": quality_issues,
        "pair_health_score": pair_health,
    }


def _mutation_work_item(pair_plan: Dict, mutation_policy: Dict, research_artifact: Dict = None, feedback: str = "") -> Dict:
    mutation_parent = pair_plan.get("mutation_parent") or {}
    mutation_axis = (
        pair_plan.get("mutation_variation_axis")
        or pair_plan.get("variation_axis")
        or pair_plan.get("bridge_axis")
        or ""
    )
    return {
        "slot": pair_plan.get("mutation_slot", 0),
        "op_type": "mutation",
        "pair_id": pair_plan.get("pair_id"),
        "parent_ids": [mutation_parent.get("id")],
        "parents": [mutation_parent],
        "mutation_parent_id": mutation_parent.get("id"),
        "mode": mutation_policy.get("mode", "hard"),
        "difficulty_label": mutation_policy.get("difficulty_label", "hard"),
        "target_diff": mutation_policy.get("target_diff", 9.0),
        "variation_axis": mutation_axis,
        "difficulty_strategy": mutation_policy.get("difficulty_strategy", "unspecified"),
        "dispatch_rationale": mutation_policy.get("rationale", ""),
        "requires_research": False,
        "research_artifact": research_artifact,
        "invariant_bundles": [bundle for bundle in [pair_plan.get("mutation_invariant_bundle")] if bundle],
        "synthesis_brief": pair_plan.get("mutation_synthesis_brief"),
        "context_pack": pair_plan.get("mutation_context_pack"),
        "context_pack_digest": dict(pair_plan.get("mutation_context_pack", {}) or {}).get("digest", ""),
        "validation_feedback": feedback,
        "mutation_policy": mutation_policy,
    }


def _normalize_variation_axis(axis: Any) -> str:
    text = str(axis or "").strip()
    if text.lower() == "axis_missing":
        return ""
    return text


def _recover_variation_axis(failed: Dict, state: AgentState, *, op_type: str) -> str:
    direct_candidates = [
        failed.get("variation_axis"),
        failed.get("variation_axis_used"),
        (failed.get("candidate_before_repair") or {}).get("variation_axis"),
        (failed.get("candidate_before_repair") or {}).get("variation_axis_used"),
    ]
    for axis in direct_candidates:
        normalized = _normalize_variation_axis(axis)
        if normalized:
            return normalized

    slot = int(failed.get("_slot", failed.get("slot", 0)) or 0)
    pair_id = str(failed.get("pair_id") or "")
    for bucket_name in ("planned_work_items", "work_items", "last_work_items"):
        for item in (state.get(bucket_name, []) or []):
            item_slot = int(item.get("slot", 0) or 0)
            item_pair_id = str(item.get("pair_id") or "")
            if item_slot != slot and (not pair_id or item_pair_id != pair_id):
                continue
            normalized = _normalize_variation_axis(item.get("variation_axis"))
            if normalized:
                return normalized

    if pair_id:
        for pair_plan in (state.get("pair_plans", []) or []):
            if str(pair_plan.get("pair_id") or "") != pair_id:
                continue
            for axis in (
                pair_plan.get("mutation_variation_axis") if op_type == "mutation" else "",
                pair_plan.get("variation_axis"),
                pair_plan.get("bridge_axis"),
            ):
                normalized = _normalize_variation_axis(axis)
                if normalized:
                    return normalized
    return ""


def _build_retry_item(failed: Dict, state: AgentState, attempts: int, repeated: bool) -> Dict:
    parents_map = {problem.get("id"): problem for problem in (state.get("current_generation", []) or [])}
    parent_ids = failed.get("parent_ids", []) or []
    parents = [parents_map[parent_id] for parent_id in parent_ids if parent_id in parents_map]
    op_type = failed.get("op_type") or ("crossover" if failed.get("type", "").startswith("crossover") else "mutation")
    reason = failed.get("constraint_guard_feedback") or failed.get("validation_feedback", "")
    reason_signature = failed.get("failure_signature") or _reason_signature(reason)
    failure_type = failed.get("failure_type", _failure_type_from_reason(reason))
    mode = failed.get("mode") or ("easy" if "easy" in (failed.get("type") or "") else "hard")
    difficulty_label = failed.get("difficulty_label") or ("easy" if mode == "easy" else "hard")
    target_diff = _coerce_difficulty_value(failed.get("difficulty", failed.get("target_diff", 9.0)), default=9.0)
    difficulty_strategy = failed.get("difficulty_strategy", "")
    variation_axis = _recover_variation_axis(failed, state, op_type=op_type)
    # Default off; the regen_planner orchestrator flips this on per slot when
    # it decides the slot should re-mine ideas before another attempt.
    requires_research = False

    if failure_type in {
        "statement_code_inconsistency",
        "statement_solution_inconsistency",
        "solvability_failure",
        "hidden_constant_failure",
        "residue_mismatch_failure",
        "no_solution_under_stated_constraints",
    }:
        repeated = repeated or attempts >= 2
        if repeated:
            mode = "easy"
            difficulty_label = "easy"
            target_diff = min(target_diff, 6.0)
            difficulty_strategy = "easier_rescue"
            requires_research = False
        rescue_note = (
            "Solvability rescue: keep the statement authoritative. "
            "Do not invent or silently reuse parent constants. "
            "Preserve exact coefficients, exponents, named equations, and derived-value dependencies unless the repaired statement explicitly and consistently changes them. "
            "If the prior coupling was inconsistent, simplify the coupling instead of overriding the statement in code or solution."
        )
        reason = f"{reason}\n{rescue_note}".strip()

    if (
        failure_type == "relation_guard_failure"
        or "target quant" in reason.lower()
        or "fundamentally different mathematical tasks" in reason.lower()
        or "sum of traces" in reason.lower()
    ):
        target_note = (
            "Target-quantity rescue: preserve the exact final target requested by the parent invariant bundle. "
            "Do not rewrite the objective into a helper statistic, trace sum, determinant count, parity count, or any other related-but-different question. "
            "The repaired child must ask for the same named final target semantics explicitly."
        )
        reason = f"{reason}\n{target_note}".strip()

    if op_type == "mutation":
        if repeated or attempts >= 2:
            mode = "easy" if mode == "hard" else "hard"
            difficulty_label = "easy" if mode == "easy" else "hard"
            target_diff = 6.0 if mode == "easy" else 9.0
    else:
        if repeated or attempts >= 2:
            if reason_signature in {"quality_gate", "near_copy", "too_easy"}:
                mode = "hard"
                difficulty_label = "hard"
                target_diff = max(target_diff, 9.0)
            else:
                mode = "easy"
                difficulty_label = "easy"
                target_diff = min(target_diff, 6.0)

    return {
        "slot": failed.get("_slot", failed.get("slot", 0)),
        "op_type": op_type,
        "retry_stage": True,
        "pair_id": failed.get("pair_id"),
        "parent_ids": parent_ids,
        "parents": parents,
        "mode": mode,
        "difficulty_label": difficulty_label,
        "target_diff": target_diff,
        "requires_research": requires_research,
        "invariant_bundles": [state.get("invariant_bundles", {}).get(parent_id) for parent_id in parent_ids if state.get("invariant_bundles", {}).get(parent_id)],
        "constraint_guard_feedback": failed.get("constraint_guard_feedback", ""),
        "validation_feedback": reason,
        "shared_invariant": failed.get("shared_invariant", ""),
        "bridge_axis": failed.get("bridge_axis", ""),
        "variation_axis": variation_axis,
        "mutation_parent_id": failed.get("mutation_parent_id"),
        "difficulty_strategy": difficulty_strategy,
        "dispatch_rationale": failed.get("dispatch_rationale", ""),
        "failure_signature": reason_signature,
        "repair_strategy": _repair_strategy_for_failure(failure_type, reason_signature),
        # Phase D3b (2026-04-16): store only the fields the repair_hotfix
        # functions actually read. The full candidate dict carried 8-12KB of
        # grounding snapshots, exploration artifacts, and internal state that
        # no downstream consumer uses.
        "candidate_before_repair": {
            k: failed.get(k)
            for k in (
                "id", "_slot", "slot", "op_type", "parent_ids",
                "statement", "answer", "solution", "code",
                "code_runtime_mode", "evidence_summary",
                "variation_axis", "variation_axis_used",
            )
            if failed.get(k) is not None
        },
    }


def _brief_key(slot: int) -> str:
    return f"slot_{slot}"


def _contract_assessment(candidate: Dict) -> Dict:
    failure_type = candidate.get("worker_failure_type", "") or ""
    failure_stage = candidate.get("worker_failure_stage", "") or ""
    failure_reason = candidate.get("worker_failure_reason", "") or ""
    if not failure_type and candidate.get("worker_status") != "ok":
        failure_type = _failure_type_from_reason(failure_reason, fallback="invariant_failure")
    return {
        "slot": candidate.get("_slot", candidate.get("slot", 0)),
        "problem_id": candidate.get("id", ""),
        "op_type": candidate.get("op_type", ""),
        "worker_status": candidate.get("worker_status", "ok"),
        "failure_type": failure_type,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "relation_guard_risk": bool(candidate.get("relation_guard_risk", False)),
        "repairable": bool(failure_type),
    }


def _repair_strategy_for_failure(failure_type: str, reason_signature: str = "") -> str:
    """Map (failure_type, reason_signature) → canonical repair strategy.

    Phase-1 Fix-#2 (2026-04-18): the post-mortem of 18 failed-repair slots
    showed three concrete patterns where the failure_type-only mapping picked
    the wrong tool:
      - `syntax_error: line 1` from code_gate_failure: the generator template
        is broken; re-running code_only_hotfix re-emits the same line-1 bug.
        Force `full_regenerate` so the template is rebuilt from scratch.
      - `not in (the )?statement` / `constraint system is inconsistent` from
        solvability_failure: the statement itself is missing the variable or
        the constraint system is unsatisfiable. code_only_hotfix preserves
        the broken statement. Force `full_regenerate`.
      - `could not extract canonical answer` keeps `code_only_hotfix` but the
        repair_brief (set in regen_planner._sanitize_decision) carries an
        explicit "print exactly one line containing only the answer" directive.
    Pattern matching is on lowercased substrings to absorb LLM phrasing drift.
    """
    sig = (reason_signature or "").lower()

    # Reason-signature overrides — checked first so failure-type defaults
    # cannot mask a more specific repair strategy.
    if "syntax_error" in sig and "line 1" in sig:
        return "full_regenerate"
    if "not in" in sig and "statement" in sig:
        return "full_regenerate"
    if "constraint system is inconsistent" in sig:
        return "full_regenerate"

    if failure_type == "missing_required_fields":
        return "full_regenerate"
    if failure_type == "sham_code_failure":
        return "code_only_hotfix"
    if failure_type == "code_gate_failure":
        return "code_only_hotfix"
    if failure_type == "target_quantity_failure":
        return "target_only_hotfix"
    if failure_type == "statement_domain_failure":
        return "statement_domain_hotfix"
    if failure_type == "relation_guard_failure":
        if reason_signature == "target_quantity":
            return "target_only_hotfix"
        if reason_signature == "statement_domain":
            return "statement_domain_hotfix"
        return "full_regenerate"
    if failure_type in {
        "statement_code_inconsistency",
        "statement_solution_inconsistency",
        "solvability_failure",
        "hidden_constant_failure",
        "residue_mismatch_failure",
    }:
        return "code_only_hotfix"
    if failure_type == "no_solution_under_stated_constraints":
        return "target_only_hotfix"
    if failure_type == "novelty_collapse_failure":
        return "full_regenerate"
    if failure_type == "invariant_failure":
        if reason_signature == "target_quantity":
            return "target_only_hotfix"
        if reason_signature == "statement_domain":
            return "statement_domain_hotfix"
        return "full_regenerate"
    if failure_type == "exploration_failure":
        return "full_regenerate"
    return "full_regenerate"


def _build_slot_failure_summary(
    failed: Dict,
    registry_entry: Dict,
    *,
    attempts: int,
    research_refetch_count: int,
    research_refetch_budget: int,
    repeated: bool,
    reason: str,
    failure_type: str,
) -> Dict[str, Any]:
    """Project one failed problem + accumulated registry into a compact
    SlotFailureSummarySchema payload the regen planner can reason over.
    """
    op_type = failed.get("op_type") or ("crossover" if str(failed.get("type", "")).startswith("crossover") else "mutation")
    mode = failed.get("mode") or ("easy" if "easy" in str(failed.get("type") or "") else "hard")
    return {
        "slot": int(failed.get("_slot", failed.get("slot", 0)) or 0),
        "op_type": op_type if op_type in {"mutation", "crossover"} else "mutation",
        "pair_id": str(failed.get("pair_id") or ""),
        "parent_ids": list(failed.get("parent_ids", []) or []),
        "failure_type": failure_type or "regenerability_failure",
        "failure_stage": str(failed.get("failure_stage") or "regenerate_failed"),
        "attempts": int(attempts),
        "research_refetch_count": int(research_refetch_count),
        "research_refetch_budget": int(research_refetch_budget),
        "repeated_signature": bool(repeated),
        "recent_failure_signatures": list(registry_entry.get("reason_signatures", []) or [])[-4:],
        "recent_failure_types": list(registry_entry.get("failure_types", []) or [])[-4:],
        "latest_failure_summary": (reason or "")[:360],
        "variation_axis": str(failed.get("variation_axis") or ""),
        "difficulty_mode": "easy" if mode == "easy" else "hard",
        "prior_decisions": list(registry_entry.get("decisions", []) or [])[-4:],
        "prior_repair_strategies": list(registry_entry.get("repair_strategies", []) or [])[-4:],
    }


def _apply_regen_decision_to_retry_item(
    retry_item: Dict[str, Any],
    decision: Dict[str, Any],
    *,
    slot: int,
    research_refetch_count: int,
    research_refetch_budget: int,
) -> Dict[str, Any]:
    """Mutate a retry_item in place based on the orchestrator decision.

    Sets ``requires_research``, injects the orchestrator's technique-only
    ``research_focus_override`` as both ``query_hint`` and a synthesis-plan
    override, and stamps the route so downstream nodes can branch.
    """
    route = decision.get("decision", "direct_repair")
    override_focus = (decision.get("research_focus_override") or "").strip()
    retry_feedback_summary = (decision.get("retry_feedback_summary") or "").strip()
    repair_strategy_override = (decision.get("repair_strategy_override") or "").strip()
    repair_brief = (decision.get("repair_brief") or "").strip()

    retry_item["_regen_decision"] = route
    retry_item["_regen_rationale"] = (decision.get("rationale") or "")[:240]
    retry_item["retry_feedback_summary"] = retry_feedback_summary
    retry_item["repair_brief"] = repair_brief
    if retry_feedback_summary:
        existing_feedback = retry_item.get("validation_feedback") or ""
        retry_item["validation_feedback"] = (
            f"{existing_feedback}\n[retry_feedback_summary] {retry_feedback_summary}".strip()
        )

    if route == "research_and_regenerate" and research_refetch_count < research_refetch_budget:
        retry_item["requires_research"] = True
        retry_item["research_refetch_count"] = research_refetch_count + 1
        if override_focus:
            retry_item["query_hint"] = override_focus
            # Seed a synthesis_plan override so the planner starts from the
            # orchestrator's technique-only focus instead of the stale focus
            # from the failed attempt.
            retry_item["synthesis_plan_override"] = {
                **(retry_item.get("synthesis_plan_override") or {}),
                "research_focus": override_focus,
            }
    else:
        retry_item["requires_research"] = False
        retry_item["research_refetch_count"] = research_refetch_count
        if route == "escalate_easier":
            retry_item["mode"] = "easy"
            retry_item["difficulty_label"] = "easy"
            retry_item["target_diff"] = min(float(retry_item.get("target_diff", 6.0) or 6.0), 6.0)
            retry_item["difficulty_strategy"] = "easier_rescue"
        if route == "giveup":
            retry_item["_regen_giveup"] = True
            retry_item["repair_strategy"] = "_giveup"
        if repair_strategy_override:
            retry_item["repair_strategy"] = repair_strategy_override

    return retry_item
