"""Regeneration node: orchestrator-centric retry stage."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage


@contextmanager
def _NULL_CTX():
    """No-op context manager — used when slot trace run is unavailable."""
    yield

from deepagent.nodes.regen.regen_planner import plan_one_slot_regen, plan_regeneration
from deepagent.tracing import lookup_slot_trace_run
from langsmith.run_helpers import tracing_context
from deepagent.tracing import build_dispatch_envelope, build_dispatch_tool_call, wrap_dispatch_result
from deepagent.state_full import AgentState

from deepagent.graph.runtime import MAX_PARALLEL_DISPATCH_FALLBACK
from deepagent.graph.artifacts import _write_json_artifact
from deepagent.graph.helpers import (
    _apply_regen_decision_to_retry_item,
    _build_retry_item,
    _build_slot_failure_summary,
    _failure_type_from_reason,
    _reason_signature,
    _record_slot_decision,
    _slot_failure_registry,
)
from deepagent.tracing import _dispatch_fifo_parallel
from deepagent.nodes.planning import _hydrate_run_working_memory

logger = logging.getLogger(__name__)


def regenerate_failed_node(state: AgentState):
    """Orchestrator-centric retry stage.

    Emits a batch `regen_planner.dispatch` tool_call envelope, invokes the
    planner LLM once over all failed slots, then applies per-slot decisions:
      * research_and_regenerate — slot goes back through synthesis_plan →
        research_candidates with a fresh technique-only research_focus;
      * direct_repair — slot goes to repair_failed_candidates with repair
        strategy (optionally overridden);
      * escalate_easier — slot is forced to easy mode, still via repair;
      * giveup — elite-backfill eventually rescues the slot.

    A per-slot `research_refetch_count` is tracked in the slot_retry_registry
    and enforced as a hard ceiling before any decision is honoured.
    """
    retry_registry = _slot_failure_registry(state)
    parameters = state.get("parameters", {}) or {}
    max_slot_regen_attempts = int(parameters.get("max_slot_regen_attempts", 2))
    research_refetch_budget = int(parameters.get("max_research_refetch", 1))

    failed_problems = state.get("failed_problems", []) or []
    # Loop guard: repair_failed_candidates → postprocess_candidates routes
    # back here when it produced no processed candidates (see
    # slot_unit). If regenerate_failed is re-entered with nothing to retry,
    # re-entering the repair loop would spin without ever incrementing the
    # validate retry counter. Short-circuit to slot_aggregate (which now owns
    # cross-slot validate decisions including elite_backfill).
    if not failed_problems:
        logger.warning(
            "regenerate_failed invoked with empty failed_problems; routing to slot_aggregate to break the repair loop."
        )
        return {
            "session_phase": "slot_aggregate",
            "work_items": [],
            "repair_queue": [],
            "failed_problems": [],
            "candidates": list(state.get("candidates", []) or []),
            "messages": [SystemMessage(content="regenerate_failed: no failed problems; delegating to slot_aggregate.")],
        }
    slot_summaries: List[Dict[str, Any]] = []
    retry_items: List[Dict] = []
    registry_updates: Dict[str, Dict[str, Any]] = {}

    for failed in failed_problems:
        slot = failed.get("_slot", 0)
        slot_key = str(slot)
        registry = dict(retry_registry.get(slot_key, {}) or {})
        previous_signatures = list(registry.get("reason_signatures", []) or [])
        reason = failed.get("constraint_guard_feedback") or failed.get("validation_feedback", "")
        signature = failed.get("failure_signature") or _reason_signature(reason)
        attempts = int(registry.get("attempts", 0)) + 1
        previous_signatures.append(signature)
        repeated = len(previous_signatures) >= 2 and previous_signatures[-1] == previous_signatures[-2]
        failure_type = failed.get("failure_type", _failure_type_from_reason(reason))
        research_refetch_count = int(registry.get("research_refetch_count", 0))
        updated_registry_entry = {
            "attempts": attempts,
            "failure_types": list((registry.get("failure_types", []) or []) + [failure_type])[-6:],
            "failure_stages": list((registry.get("failure_stages", []) or []) + [failed.get("failure_stage", "regenerate_failed")])[-6:],
            "reason_signatures": previous_signatures[-4:],
            "research_refetch_count": research_refetch_count,
            "last_reason": reason,
            "last_failure_type": failure_type,
            "last_failure_stage": failed.get("failure_stage", "regenerate_failed"),
            "last_repair_strategy": failed.get("repair_strategy", ""),
        }
        registry_updates[slot_key] = updated_registry_entry
        retry_item = _build_retry_item(
            failed,
            state,
            attempts=attempts,
            repeated=repeated or attempts >= max_slot_regen_attempts,
        )
        # Carry the refetch counter on the item so downstream nodes (e.g. the
        # next regenerate_failed cycle) can read it without the registry.
        retry_item["research_refetch_count"] = research_refetch_count
        retry_items.append(retry_item)
        slot_summaries.append(
            _build_slot_failure_summary(
                failed,
                updated_registry_entry,
                attempts=attempts,
                research_refetch_count=research_refetch_count,
                research_refetch_budget=research_refetch_budget,
                repeated=repeated,
                reason=reason,
                failure_type=failure_type,
            )
        )

    # ── Orchestrator dispatch envelope ───────────────────────────────────
    from prompts import RegenPlanDispatchArgs  # local import keeps the module
    generation_count = int(state.get("generation_count", 0))
    retry_round = 1 + max(
        int((retry_registry.get(str(s.get("slot", 0)), {}) or {}).get("retry_round", 0))
        for s in slot_summaries
    ) if slot_summaries else 1
    dispatch_messages: List[Any] = []
    plan: Dict[str, Any] = {"per_slot_decisions": [], "overall_rationale": "No failed slots."}
    tool_call_id = ""
    if slot_summaries:
        dispatch_args = RegenPlanDispatchArgs.model_validate(
            {
                "generation_count": generation_count,
                "retry_round": retry_round,
                "research_refetch_budget": research_refetch_budget,
                "max_slot_regen_attempts": max_slot_regen_attempts,
                "failed_slots": slot_summaries,
            }
        ).model_dump()
        tool_call = build_dispatch_tool_call("regen_planner.dispatch", dispatch_args)
        tool_call_id = tool_call["id"]
        dispatch_messages.append(
            build_dispatch_envelope(
                [tool_call],
                content=f"Dispatching {len(slot_summaries)} failed slots to regen_planner.",
            )
        )
        # Phase E: per-slot regen_plan. Each slot's LLM call runs INSIDE
        # its own `generator.operation.slot_X` trace span (looked up via
        # the slot trace registry) so the regen decision shows as a child
        # of the slot's lifecycle, not floating at the orchestrator level.
        # The slot trace span stays open across attempts (Phase E.1
        # reverted the per-attempt close), so by the time regen runs after
        # validate failure, the slot's span is still active and serves as
        # the parent for these per-slot regen LLM calls.
        slot_trace_state_for_lookup = {
            "session_thread_id": str(state.get("session_thread_id", "") or ""),
            "generation_count": generation_count,
        }
        per_slot_decisions: List[Dict[str, Any]] = []
        for summary in slot_summaries:
            slot_id = int(summary.get("slot", 0) or 0)
            slot_run = lookup_slot_trace_run(slot_trace_state_for_lookup, slot_id)
            invoke_config_one = {
                "run_name": f"deepagent.regen_planner.slot_{slot_id}",
                "tags": ["deepagent", "regen_planner", f"slot_{slot_id}"],
                "metadata": {
                    "generation_count": generation_count,
                    "retry_round": retry_round,
                    "slot": slot_id,
                    "dispatch_tool_call_id": tool_call_id,
                },
            }
            # If we found the slot's trace span, parent the LLM call under
            # it; otherwise the LLM falls back to the ambient parent
            # (orchestrator.regenerate_failed wrapper).
            ctx = tracing_context(parent=slot_run) if slot_run is not None else _NULL_CTX()
            with ctx:
                decision_one = plan_one_slot_regen(
                    summary,
                    generation_count=generation_count,
                    retry_round=retry_round,
                    research_refetch_budget=research_refetch_budget,
                    max_slot_regen_attempts=max_slot_regen_attempts,
                    invoke_config=invoke_config_one,
                )
            per_slot_decisions.append(decision_one)
        plan = {
            "per_slot_decisions": per_slot_decisions,
            "overall_rationale": f"Per-slot regen: {len(per_slot_decisions)} slot(s) routed independently.",
        }
        dispatch_messages.append(
            wrap_dispatch_result(
                tool_call_id,
                "regen_planner.dispatch",
                {
                    "retry_round": retry_round,
                    "decisions": [
                        {"slot": d["slot"], "decision": d["decision"]}
                        for d in (plan.get("per_slot_decisions", []) or [])
                    ],
                    "overall_rationale": plan.get("overall_rationale", ""),
                },
            )
        )

    # ── Apply decisions to retry items ────────────────────────────────────
    decisions_by_slot = {int(d.get("slot", 0) or 0): d for d in (plan.get("per_slot_decisions", []) or [])}
    research_items: List[Dict] = []
    repair_items: List[Dict] = []
    giveup_items: List[Dict] = []
    for retry_item in retry_items:
        slot = int(retry_item.get("slot", 0) or 0)
        slot_key = str(slot)
        decision = decisions_by_slot.get(slot) or {
            "decision": "direct_repair",
            "research_focus_override": "",
            "repair_strategy_override": "",
            "retry_feedback_summary": "",
            "rationale": "No decision returned; default to direct_repair.",
        }
        research_refetch_count = int(registry_updates.get(slot_key, {}).get("research_refetch_count", 0))
        _apply_regen_decision_to_retry_item(
            retry_item,
            decision,
            slot=slot,
            research_refetch_count=research_refetch_count,
            research_refetch_budget=research_refetch_budget,
        )
        # Record the applied decision so the next regen cycle's planner sees
        # the routing history and can detect ineffective repeats.
        _record_slot_decision(registry_updates, slot, str(decision.get("decision", "")))
        if retry_item.get("_regen_giveup"):
            # Giveup slots bypass the repair/research queue entirely. Their
            # original failed_problem record is preserved below so the next
            # validate cycle (with bumped retry counter) routes them to
            # elite_backfill instead of looping.
            giveup_items.append(retry_item)
            continue
        if retry_item.get("requires_research"):
            # Persist the bumped refetch counter back to the registry.
            registry_updates[slot_key]["research_refetch_count"] = int(retry_item.get("research_refetch_count", research_refetch_count + 1))
            research_items.append(retry_item)
        else:
            repair_items.append(retry_item)

    # Preserve the failed_problem records for giveup slots so they remain
    # visible to validate_candidates → elite_backfill on the next cycle.
    giveup_slot_keys = {int(item.get("slot", 0) or 0) for item in giveup_items}
    preserved_failed = [
        fp for fp in failed_problems
        if int(fp.get("_slot", fp.get("slot", 0)) or 0) in giveup_slot_keys
    ]

    retry_registry_after = {**retry_registry, **registry_updates}

    # ── Hydrate + route ──────────────────────────────────────────────────
    research_session = bool(research_items)
    # Hydrate both research and repair items together so both get full context.
    # Previously only one set was hydrated, which caused repair_items to be
    # silently dropped (set to []) when research_session=True.
    all_hydrate_items = research_items + repair_items
    hydrated_all, hydrated_pair_plans, run_working_memory, synthesis_briefs, research_skipped_slots, llm_brief_skipped_slots = _hydrate_run_working_memory(
        {**state, "slot_retry_registry": retry_registry_after, "slot_failure_registry": retry_registry_after},
        all_hydrate_items,
        state.get("pair_plans", []) or [],
        state.get("selection_strategy", {}) or {},
    )

    # Split hydrated items back into their destination buckets.
    research_slot_keys = {int(item.get("slot", 0) or 0) for item in research_items}
    hydrated_research = [x for x in hydrated_all if int(x.get("slot", 0) or 0) in research_slot_keys]
    hydrated_repair = [x for x in hydrated_all if int(x.get("slot", 0) or 0) not in research_slot_keys]

    # When ALL slots are given up, force elite_backfill immediately by bumping
    # _validation_retry_count past max_validation_retries. When SOME slots are
    # given up but others are still in active repair/research, do NOT bump —
    # bumping would prematurely terminate the still-active slots' retry budget
    # via the shared counter. Giveup slots will instead propagate as
    # failed_problems through the natural validate cycles until elite_backfill
    # fires for them.
    params_out = dict(parameters)
    all_giveup = bool(giveup_items) and not research_items and not repair_items
    if all_giveup:
        max_validation_retries = int(params_out.get("max_validation_retries", 2))
        params_out["_validation_retry_count"] = max(
            int(params_out.get("_validation_retry_count", 0)),
            max_validation_retries + 1,
        )

    if all_giveup:
        summary_text = (
            f"Orchestrator gave up on all {len(giveup_items)} slot(s); "
            f"routing directly to slot_aggregate for elite_backfill rescue."
        )
        dispatch_messages.append(SystemMessage(content=summary_text))
        result = {
            "session_phase": "slot_aggregate",
            "work_items": [],
            "repair_queue": [],
            "failed_problems": preserved_failed,
            "candidates": list(state.get("candidates", []) or []),
            "retry_registry": retry_registry_after,
            "slot_retry_registry": retry_registry_after,
            "slot_failure_registry": retry_registry_after,
            "regen_plan": plan,
            "parameters": params_out,
            "messages": dispatch_messages,
        }
        _write_json_artifact(
            {**state, **result},
            "regenerate_failed",
            {
                "generation_count": generation_count,
                "retry_items": [],
                "giveup_items": giveup_items,
                "regen_plan": plan,
                "retry_round": retry_round,
                "research_refetch_budget": research_refetch_budget,
            },
        )
        return result

    # Routing (Option A repair-fan-out integration):
    #   research_session  → "synthesis_plan"        (research items go through
    #                        research_candidates → prepare_synthesis_briefs →
    #                        slot_dispatch → slot_unit; repair items piggyback
    #                        in work_items so slot_unit handles them too).
    #   else              → "synthesize_candidates" (routes directly to
    #                        slot_dispatch; slot_unit detects `repair_strategy`
    #                        on each item and dispatches to _run_repair_item).
    # repair_failed_candidates_node is no longer reached on either path.
    if research_session:
        next_phase = "synthesis_plan"
        # Carry repair items alongside research items so they fan-out together.
        work_items_out = hydrated_research + hydrated_repair
        summary_text = (
            f"Orchestrator routed {len(research_items)} slot(s) through research and "
            f"{len(repair_items)} slot(s) for direct repair (both via slot fan-out); "
            f"{len(giveup_items)} slot(s) given up."
        )
    else:
        next_phase = "synthesize_candidates"  # routes to slot_dispatch
        work_items_out = hydrated_repair
        summary_text = (
            f"Orchestrator routed all {len(repair_items)} slot(s) to repair via slot fan-out; "
            f"{len(giveup_items)} slot(s) given up."
        )
    dispatch_messages.append(SystemMessage(content=summary_text))

    result = {
        "session_phase": next_phase,
        "work_items": work_items_out,
        "pair_plans": hydrated_pair_plans,
        "run_working_memory": run_working_memory,
        "retry_registry": retry_registry_after,
        "slot_retry_registry": retry_registry_after,
        "slot_failure_registry": retry_registry_after,
        "synthesis_briefs": synthesis_briefs,
        "context_packs": dict(run_working_memory.get("context_packs", {}) or {}),
        "research_skipped_slots": research_skipped_slots,
        "llm_brief_skipped_slots": llm_brief_skipped_slots,
        "fast_path_slot_count": len(llm_brief_skipped_slots),
        "failed_problems": preserved_failed,
        "candidates": [],
        "repair_queue": [],  # drained — repair items now travel via work_items
        "regen_plan": plan,
        "parameters": params_out,
        "messages": dispatch_messages,
    }
    _write_json_artifact(
        {**state, **result},
        "regenerate_failed",
        {
            "generation_count": generation_count,
            "retry_items": hydrated_all,
            "regen_plan": plan,
            "retry_round": retry_round,
            "research_refetch_budget": research_refetch_budget,
        },
    )
    return result
