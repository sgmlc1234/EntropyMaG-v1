"""Synthesis-phase nodes: synthesize_candidates, repair_failed_candidates, and pair pipeline."""

import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Tuple

from langchain_core.messages import SystemMessage
from langsmith.run_helpers import get_current_run_tree

from deepagent.nodes.synthesis.generator import deep_crossover_problems, deep_mutate_problem
from deepagent.nodes.synthesis.repair_hotfix import apply_code_only_hotfix, apply_statement_domain_hotfix, apply_target_only_hotfix
from deepagent.tracing import build_dispatch_envelope, build_dispatch_tool_call, digest_payload, wrap_dispatch_result
from deepagent.state_full import AgentState

from deepagent.graph.runtime import (
    MAX_PARALLEL_DISPATCH_FALLBACK,
    _coerce_difficulty_value,
    _desired_generation_size,
)
from deepagent.graph.artifacts import (
    _slim_problem, _slim_failed_problem, _slim_work_item, _write_json_artifact,
)
from deepagent.graph.helpers import (
    _all_invariant_flags_true,
    _contract_assessment,
    _derive_child_lineage_metrics,
    _determine_mutation_policy,
    _ensure_unique_problem_id,
    _failure_type_from_reason,
    _lineage_metrics_for_problem,
    _make_failed_problem,
    _mutation_work_item,
    _reason_signature,
    _record_slot_failure,
    _repair_strategy_for_failure,
    _slot_failure_registry,
)
from deepagent.tracing import (
    _dispatch_fifo_parallel,
    _run_with_child_trace,
    _trace_identity,
    ensure_slot_trace_parent,
)

logger = logging.getLogger(__name__)


def _problem_slot(problem: Dict[str, Any]) -> int:
    try:
        return int(problem.get("_slot", problem.get("slot", 0)) or 0)
    except Exception:
        return 0


def _problem_peer_summary(problem: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "slot": _problem_slot(problem),
        "problem_id": problem.get("id", ""),
        "op_type": problem.get("op_type", problem.get("type", "")),
        "pair_id": problem.get("pair_id"),
        "difficulty": problem.get("difficulty"),
        "variation_axis": str(problem.get("variation_axis_used") or problem.get("variation_axis") or "")[:120],
        "statement_preview": str(problem.get("statement", "") or "")[:180],
        "answer_preview": str(problem.get("answer", "") or "")[:80],
    }


def _attach_peer_context(problems: List[Dict[str, Any]], *, phase: str, failed_problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    peer_summaries = [_problem_peer_summary(problem) for problem in problems]
    failed_summaries = [
        {
            "slot": _problem_slot(problem),
            "problem_id": problem.get("id", ""),
            "failure_type": problem.get("failure_type", ""),
            "failure_stage": problem.get("failure_stage", ""),
            "failure_reason": str(problem.get("failure_reason", problem.get("validation_feedback", "")) or "")[:180],
        }
        for problem in (failed_problems or [])
    ]
    enriched: List[Dict[str, Any]] = []
    for problem in problems:
        slot = _problem_slot(problem)
        updated = dict(problem)
        updated["orchestrator_peer_context"] = {
            "phase": phase,
            "peer_count": max(0, len(peer_summaries) - 1),
            "peer_summaries": [summary for summary in peer_summaries if summary.get("slot") != slot][:3],
            "failed_slot_summaries": [summary for summary in failed_summaries if summary.get("slot") != slot][:3],
        }
        enriched.append(updated)
    return enriched


def _run_repair_item(state: AgentState, item: Dict[str, Any], slot_failure_registry: Dict[str, Any]) -> Dict[str, Any]:
    try:
        slot = item.get("slot", 0)
        op_type = item.get("op_type")
        parents = item.get("parents", [])
        repair_strategy = item.get("repair_strategy", "") or "full_regenerate"
        invoke_config = {
            "run_name": f"deepagent.repair.{repair_strategy}.{op_type}.slot_{slot}",
            "tags": ["deepagent", "generator", op_type, "repair"],
            "metadata": {
                "slot": slot,
                "pair_id": item.get("pair_id"),
                "op_type": op_type,
                "repair_strategy": repair_strategy,
                "normalization_mode": item.get("normalization_mode", ""),
                "desired_generation_size": item.get("desired_generation_size"),
                "target_generation_size": item.get("target_generation_size"),
                "current_generation_size": item.get("current_generation_size"),
                "strategy_source": item.get("strategy_source", ""),
            },
        }
        # Surface accumulated retry history to the repair worker. Without this,
        # each repair call sees only the latest validation_feedback — repeated
        # identical-signature failures look like a fresh attempt every time and
        # the repair LLM keeps trying the same fix. With it, the worker can
        # detect "I've already failed this way 3 times, try something else".
        slot_reg = slot_failure_registry.get(str(slot), {}) or {}
        attempts = int(slot_reg.get("attempts", 0) or 0)
        prior_block = ""
        if attempts > 1:
            prior_types = list(slot_reg.get("failure_types", []) or [])[-4:]
            prior_sigs = list(slot_reg.get("reason_signatures", []) or [])[-4:]
            prior_block = (
                f"\n[prior_attempts] attempt={attempts} "
                f"failure_types={prior_types} signatures={prior_sigs}"
            )
        # repair_brief is the planner's structured "try this differently"
        # synthesis (Phase 4 / R-1+R-4). Required when attempts >= 3 and the
        # planner did not give up, optional otherwise.
        brief = str(item.get("repair_brief", "") or "").strip()
        brief_block = f"\n[repair_brief] {brief}" if brief else ""
        repair_feedback = (
            f"[repair_strategy={repair_strategy}]"
            f"{prior_block}"
            f"{brief_block}\n"
            f"{item.get('validation_feedback', '')}"
        ).strip()
        # When the regen orchestrator escalates a slot to easier mode, surface
        # that signal to the hotfix workers so they can relax constraints
        # instead of re-applying the same hard-mode rigor that already failed.
        escalation_signal = (
            "escalate_easier" if item.get("difficulty_strategy") == "easier_rescue" else ""
        )
        base_candidate = dict(item.get("candidate_before_repair") or {})
        if repair_strategy == "code_only_hotfix":
            registry_entry = slot_failure_registry.get(str(slot)) or {}
            recent_types = list((registry_entry.get("failure_types") or [])[-4:])
            cycle_count = sum(
                1 for i in range(len(recent_types) - 1)
                if recent_types[i] in {"sham_code_failure", "code_gate_failure"}
                and recent_types[i + 1] in {"invariant_failure", "code_gate_failure"}
            )
            if cycle_count >= 2:
                logger.warning(
                    "slot %s: sham→hotfix→invariant cycle detected (%d times); upgrading code_only_hotfix → full_regenerate.",
                    slot,
                    cycle_count,
                )
                repair_strategy = "full_regenerate"
                item["repair_strategy"] = "full_regenerate"
            if repair_strategy == "code_only_hotfix":
                if not base_candidate:
                    raise ValueError("Code-only hotfix requires candidate_before_repair.")
                child = apply_code_only_hotfix(
                    base_candidate,
                    invariant_bundles=item.get("invariant_bundles"),
                    synthesis_brief=item.get("synthesis_brief"),
                    validation_feedback=repair_feedback,
                    escalation_signal=escalation_signal,
                    invoke_config=invoke_config,
                )
            elif repair_strategy == "target_only_hotfix":
                if not base_candidate:
                    raise ValueError("Target-only hotfix requires candidate_before_repair.")
                child = apply_target_only_hotfix(
                    base_candidate,
                    invariant_bundles=item.get("invariant_bundles"),
                    synthesis_brief=item.get("synthesis_brief"),
                    validation_feedback=repair_feedback,
                    escalation_signal=escalation_signal,
                    invoke_config=invoke_config,
                )
            elif repair_strategy == "statement_domain_hotfix":
                if not base_candidate:
                    raise ValueError("Statement-domain hotfix requires candidate_before_repair.")
                child = apply_statement_domain_hotfix(
                    base_candidate,
                    invariant_bundles=item.get("invariant_bundles"),
                    synthesis_brief=item.get("synthesis_brief"),
                    validation_feedback=repair_feedback,
                    escalation_signal=escalation_signal,
                    invoke_config=invoke_config,
                )
        elif op_type == "mutation" and parents:
            child = deep_mutate_problem(
                parents[0],
                mode=item.get("mode", "hard"),
                difficulty_label=item.get("difficulty_label", "hard"),
                target_diff=item.get("target_diff", 9.0),
                research_artifact=item.get("research_artifact"),
                invariant_bundles=item.get("invariant_bundles"),
                variation_axis=item.get("variation_axis", ""),
                difficulty_strategy=item.get("difficulty_strategy", "harder_exploratory"),
                dispatch_rationale=item.get("dispatch_rationale", ""),
                synthesis_brief=item.get("synthesis_brief"),
                context_pack=item.get("context_pack"),
                invoke_config=invoke_config,
                validation_feedback=repair_feedback,
            )
        elif op_type == "crossover" and len(parents) >= 2:
            child = deep_crossover_problems(
                parents[0],
                parents[1],
                mode=item.get("mode", "hard"),
                difficulty_label=item.get("difficulty_label", "hard"),
                target_diff=item.get("target_diff", 9.0),
                research_artifact=item.get("research_artifact"),
                invariant_bundles=item.get("invariant_bundles"),
                variation_axis=item.get("variation_axis", ""),
                difficulty_strategy=item.get("difficulty_strategy", "harder_exploratory"),
                dispatch_rationale=item.get("dispatch_rationale", ""),
                synthesis_brief=item.get("synthesis_brief"),
                context_pack=item.get("context_pack"),
                invoke_config=invoke_config,
                validation_feedback=repair_feedback,
            )
        else:
            raise ValueError(f"Unsupported repair work item: {op_type}")
        child["_slot"] = slot
        child["pair_id"] = item.get("pair_id")
        child["op_type"] = op_type
        child["generation_meta"] = {
            "generation_count": state.get("generation_count", 0),
            "slot": slot,
            "op_type": op_type,
            "pair_id": item.get("pair_id"),
            "context_pack_digest": item.get("context_pack_digest", ""),
            "repair_strategy": repair_strategy,
        }
        child["lineage_metrics"] = _derive_child_lineage_metrics(parents, op_type)
        return {"candidate": child, "error": ""}
    except Exception as exc:
        return {"candidate": None, "error": str(exc)}


def _build_synthesis_dispatch_args(work_item: Dict) -> Dict:
    """Project a work_item into a validated SynthesisDispatchArgs payload."""
    from prompts import SynthesisDispatchArgs
    op_type = work_item.get("op_type", "mutation")
    if op_type == "survivor":
        raise ValueError("synthesize dispatch is not valid for survivor slots")
    mode = work_item.get("mode") or "hard"
    if mode == "carry":
        mode = "hard"
    difficulty_label = work_item.get("difficulty_label") or ("hard" if mode == "hard" else "easy")
    if difficulty_label not in {"easy", "medium", "hard", "superhard"}:
        difficulty_label = "hard"
    target_diff = float(work_item.get("target_diff", 0.0) or 0.0)
    if target_diff <= 1.0:
        target_diff = 9.0 if mode == "hard" else 6.0
    brief = dict(work_item.get("synthesis_brief") or {})
    plan = dict(work_item.get("synthesis_plan") or {})
    payload = {
        "slot": int(work_item.get("slot", 0) or 0),
        "pair_id": str(work_item.get("pair_id", "") or ""),
        "op_type": op_type,
        "parent_ids": list(work_item.get("parent_ids", []) or [parent.get("id", "") for parent in (work_item.get("parents", []) or [])]),
        "mode": mode,
        "difficulty_label": difficulty_label,
        "target_diff": target_diff,
        "variation_axis": str(work_item.get("variation_axis", "") or ""),
        "preferred_composition_pattern": str(
            brief.get("preferred_composition_pattern")
            or plan.get("preferred_composition_pattern")
            or ""
        ),
        "synthesis_brief_digest": digest_payload(brief) if brief else "",
    }
    return SynthesisDispatchArgs.model_validate(payload).model_dump()

