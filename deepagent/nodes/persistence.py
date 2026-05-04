"""Persistence nodes: save_generation and consolidate_archive."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage

from artifact_views import write_all_validated_problem_views, write_latest_problem_views
from data_paths import DEFAULT_GENERATION_FORMAT, ensure_parent_dir
from deepagent.invariants import extract_invariant_bundles
from deepagent.memory_bank import consolidate_run_to_archive
from deepagent.state_full import AgentState

from deepagent.graph.runtime import (
    _build_review_prompt,
    _canonical_target_generation_size,
    _cumulative_validated_generated_count,
    _desired_generation_size,
    _normalized_parameters,
)
from deepagent.graph.artifacts import _write_json_artifact
from deepagent.graph.helpers import _lineage_index
from deepagent.tracing import _dispatch_fifo_parallel, close_slot_trace_parents, record_deterministic_span


def _problem_slot(problem: Dict[str, Any]) -> int:
    try:
        return int(problem.get("_slot", problem.get("slot", 0)) or 0)
    except Exception:
        return 0


def save_generation_node(state: AgentState):
    approved = state.get("approved_candidates", []) or []
    ordered = [item["problem"] for item in sorted(approved, key=lambda item: item.get("slot", 0))]
    required_population = _desired_generation_size(state)
    params = _normalized_parameters(state)
    min_survivable_population = max(1, min(int(params.get("min_survivable_population", 3) or 3), required_population))
    if len(ordered) < min_survivable_population:
        raise RuntimeError(
            f"Refusing to save undersized generation: {len(ordered)}/{required_population} approved "
            f"with minimum survivable population {min_survivable_population}."
        )
    # Snapshot plan-outcome inputs BEFORE we strip internal fields. This bridges
    # the state gap between save_generation (which clears work_items for the
    # next generation) and consolidate_archive (which needs the prior-generation
    # work_items + slot mapping to build plan_outcome_cards).
    saved_slot_map: Dict[int, str] = {}
    for problem in ordered:
        slot = problem.get("_slot", problem.get("slot"))
        if slot is None:
            continue
        try:
            saved_slot_map[int(slot)] = str(problem.get("id", "") or "")
        except Exception:
            continue
    # Prefer the stable planned_work_items snapshot (set at synthesis_plan_node
    # and preserved through regenerate loops). Fall back to the live work_items
    # for generations where synthesis_plan was skipped (all-survivor plans).
    last_work_items_snapshot = [
        dict(item)
        for item in (state.get("planned_work_items") or state.get("work_items") or [])
    ]
    last_failed_problems_snapshot = [dict(p) for p in (state.get("failed_problems", []) or [])]
    # Phase D3c (2026-04-16): expanded orchestrator-internal field strip.
    # Fields matching these prefixes/names are pipeline bookkeeping, not
    # part of the user-facing problem object.
    pre_save_problems = [dict(problem) for problem in ordered]
    save_trace_items: List[Dict[str, Any]] = []
    for problem in pre_save_problems:
        slot = _problem_slot(problem)
        if (problem.get("type") or "").lower() in {"survivor", "fallback_survivor"}:
            continue
        save_trace_items.append(
            {
                **problem,
                "slot": slot,
                "_slot_trace_enabled": True,
                "_trace_name": f"orchestrator.save_generation.slot_{slot}",
            }
        )

    def _save_trace_worker(problem: Dict[str, Any]) -> Dict[str, Any]:
        peer_context = dict(problem.get("orchestrator_peer_context") or {})
        record_deterministic_span(
            "orchestrator.save_generation.finalize",
            inputs={
                "slot": _problem_slot(problem),
                "problem_id": problem.get("id", ""),
                "op_type": problem.get("op_type", problem.get("type", "")),
                "pair_id": problem.get("pair_id"),
            },
            outputs={
                "status": "ready_to_persist",
                "peer_count": int(peer_context.get("peer_count", 0) or 0),
                "peer_problem_ids": [summary.get("problem_id", "") for summary in (peer_context.get("peer_summaries") or [])[:3]],
                "failed_peer_slots": [summary.get("slot") for summary in (peer_context.get("failed_slot_summaries") or [])[:3]],
            },
            tags=["deepagent", "save_generation"],
            metadata={"slot": _problem_slot(problem), "problem_id": problem.get("id", "")},
        )
        return {"status": "ready_to_persist"}

    if save_trace_items:
        _dispatch_fifo_parallel(save_trace_items, int(params.get("max_parallel_dispatch", 1) or 1), _save_trace_worker, slot_trace_state=state)

    _SAVE_STRIP_FIELDS = (
        "_slot",
        "_code_execution_assessment",
        "_regen_decision",
        "_regen_rationale",
        "_regen_giveup",
        "_dispatch_args",
        "_dispatch_tool_call_id",
        "_trace_name",
        "_slot_trace_enabled",
        "_slot_trace_name",
        "_slot_trace_tags",
        "_slot_trace_metadata",
        "invariant_audit",
        "constraint_guard_assessment",
        "worker_answer",
        "answer_materialization_reason",
        "candidate_before_repair",
        "exploration_artifact",
        "orchestrator_materialized_answer",
        "orchestrator_peer_context",
        "ground_and_rescore_assessment",
        "pair_quality_assessment",
    )
    for problem in ordered:
        for key in _SAVE_STRIP_FIELDS:
            problem.pop(key, None)
    gen_count = state.get("generation_count", 0)
    gen_num = gen_count + 1
    filename = params.get("save_format", DEFAULT_GENERATION_FORMAT)
    try:
        filename = filename.format(gen=gen_num)
    except Exception:
        filename = DEFAULT_GENERATION_FORMAT.format(gen=gen_num)
    ensure_parent_dir(filename)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)
    slot_outputs: Dict[int, Dict[str, Any]] = {}
    for problem in pre_save_problems:
        slot = _problem_slot(problem)
        slot_outputs[slot] = {
            "status": "saved",
            "problem_id": problem.get("id", ""),
            "output_file": filename,
            "peer_count": int(((problem.get("orchestrator_peer_context") or {}).get("peer_count", 0)) or 0),
        }
    for failed_problem in (state.get("failed_problems", []) or []):
        slot = _problem_slot(failed_problem)
        slot_outputs.setdefault(
            slot,
            {
                "status": "not_saved",
                "problem_id": failed_problem.get("id", ""),
                "failure_stage": failed_problem.get("failure_stage", ""),
                "failure_type": failed_problem.get("failure_type", ""),
            },
        )
    close_slot_trace_parents(state, slot_outputs=slot_outputs, default_status="not_saved")
    artifact_dir = Path((state.get("session_options", {}) or {}).get("artifact_dir") or Path(filename).parent)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_latest_problem_views(artifact_dir)
    write_all_validated_problem_views(artifact_dir)
    # Flag problems with a failed quality gate so downstream selectors can
    # deprioritize or swap them as parents. quality_assessment is not in the
    # strip list, but adding an explicit boolean avoids nested-dict lookups.
    for problem in ordered:
        qa = problem.get("quality_assessment") or {}
        if not qa.get("passed", True):
            problem["quality_low"] = True
    non_survivor_count = sum(1 for problem in ordered if problem.get("type") not in {"survivor", "fallback_survivor"})
    cumulative_generated_count = _cumulative_validated_generated_count(state) + non_survivor_count
    # P3.2: append the saved survivor's answer to the ring buffer so the next
    # generation's selector can detect collapse chains. Buffer is capped at 3.
    # W5: exclude fallback_survivor — those are emergency elite-backfills, not
    # genuine survivor picks, and counting them would force unnecessary swaps.
    prior_recent_answers = list(state.get("recent_survivor_answers", []) or [])
    saved_survivor = next(
        (p for p in ordered if (p.get("type") or "").lower() == "survivor"),
        None,
    )
    if saved_survivor and str(saved_survivor.get("answer", "")).strip():
        prior_recent_answers.append(str(saved_survivor.get("answer", "")).strip())
    recent_survivor_answers = prior_recent_answers[-3:]
    result = {
        "current_generation": ordered,
        "current_generation_size": len(ordered),
        "generation_save_status": "partial_save" if len(ordered) < required_population else "complete",
        "generation_count": gen_num,
        "desired_generation_size": len(ordered),
        "target_generation_size": _canonical_target_generation_size(state.get("parameters", {}) or {}),
        "cumulative_validated_generated_count": cumulative_generated_count,
        "stop_reason": "",
        "session_phase": "consolidate_archive",
        "awaiting_review": False,
        "review_prompt": _build_review_prompt(
            {
                **state,
                "generation_count": gen_num,
                "current_generation": ordered,
                "current_generation_size": len(ordered),
                "cumulative_validated_generated_count": cumulative_generated_count,
                "session_options": state.get("session_options", {}) or {},
            }
        ),
        "approved_candidates": [],
        "candidates": [],
        "work_items": [],
        "pair_plans": [],
        "pair_results": {},
        "pair_health_scores": {},
        "mutation_policies": {},
        "synthesis_briefs": {},
        # Snapshots consumed by consolidate_archive to build plan_outcome_cards.
        # These are cleared (emptied) at the next generation's init_run_memory.
        "last_work_items": last_work_items_snapshot,
        "last_failed_problems": last_failed_problems_snapshot,
        "last_saved_slot_map": saved_slot_map,
        "run_working_memory": {
            **(state.get("run_working_memory", {}) or {}),
            "accepted_candidate_deltas": [
                {
                    "id": problem.get("id", ""),
                    "variation_axis_used": problem.get("variation_axis_used", ""),
                    "difficulty_strategy": problem.get("difficulty_strategy", ""),
                }
                for problem in ordered
                if problem.get("type") not in {"survivor", "fallback_survivor"}
            ],
        },
        "context_packs": {},
        "lineage_metrics": _lineage_index(ordered),
        "validation_feedback": [],
        "repair_queue": [],
        "retry_registry": {},
        "slot_retry_registry": {},
        "slot_failure_registry": {},
        "validation_evidence": {},
        "generation_handoff_ready": False,
        "recent_survivor_answers": recent_survivor_answers,
        "messages": [SystemMessage(content=f"Saved generation {gen_num} to {filename}")],
    }
    _write_json_artifact(
        {**state, **result},
        "save_generation",
        {
            "generation_count": gen_num,
            "output_file": filename,
            "current_generation": ordered,
            "pair_results": state.get("pair_results", {}),
            "pair_health_scores": state.get("pair_health_scores", {}),
            "mutation_policies": state.get("mutation_policies", {}),
        },
    )
    return result


def consolidate_archive_node(state: AgentState):
    current_generation = state.get("current_generation", []) or []
    invariant_bundles = extract_invariant_bundles(current_generation)
    # save_generation_node clears work_items/failed_problems for the next
    # generation, so consolidate must fall back to the snapshots it left behind.
    work_items = list(
        state.get("work_items")
        or state.get("last_work_items")
        or []
    )
    selection_strategy = dict(state.get("selection_strategy", {}) or {})
    failed_problems = list(
        state.get("failed_problems")
        or state.get("last_failed_problems")
        or []
    )
    saved_slot_map = dict(state.get("last_saved_slot_map", {}) or {})
    archive = consolidate_run_to_archive(
        current_generation=current_generation,
        generation_count=state.get("generation_count", 0),
        session_options=state.get("session_options", {}) or {},
        invariant_bundles=invariant_bundles,
        failed_problems=failed_problems,
        archival_memory_handle=state.get("archival_memory_handle", {}) or {},
        selection_strategy=selection_strategy,
        work_items=work_items,
        saved_slot_map=saved_slot_map,
    )
    plan_outcome_cards = list(archive.get("plan_outcome_cards", []) or [])
    result = {
        "session_phase": "review_generation",
        "archival_memory_handle": archive,
        "invariant_bundles": invariant_bundles,
        "messages": [SystemMessage(content="Consolidated saved generation into archival memory.")],
    }
    _write_json_artifact(
        {**state, **result},
        "consolidate_archive",
        {
            "generation_count": state.get("generation_count", 0),
            "archive_metrics": archive.get("metrics", {}),
            "archive_paths": archive.get("paths", {}),
        },
    )
    _write_json_artifact(
        {**state, **result},
        "plan_outcome",
        {
            "generation_count": state.get("generation_count", 0),
            "strategy_source": selection_strategy.get("strategy_source", ""),
            "plan_outcome_cards_this_generation": [
                card for card in plan_outcome_cards
                if int(card.get("generation", 0) or 0) == int(state.get("generation_count", 0) or 0)
            ],
            "plan_outcome_cards_window": plan_outcome_cards,
        },
    )
    return result
