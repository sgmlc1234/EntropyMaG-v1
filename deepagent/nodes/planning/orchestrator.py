"""Planning-phase nodes: plan_generation, synthesis_plan, research, briefs."""

import logging
from typing import Any, Dict, List, Tuple

from langchain_core.messages import SystemMessage

from deepagent.nodes.planning.briefing import prepare_synthesis_brief
from deepagent.family_strategy import should_use_llm_brief, should_use_research
from deepagent.invariants import extract_invariant_bundles
from deepagent.memory_bank import build_run_working_memory, summarize_plan_outcome_cards
from deepagent.nodes.planning.researcher import build_research_dispatch_args, research_work_item
from deepagent.nodes.planning.selector import plan_generation
from deepagent.tracing import build_dispatch_envelope, build_dispatch_tool_call, wrap_dispatch_result
from deepagent.state_full import AgentState
from deepagent.nodes.planning.synthesis_planner import build_synthesis_plan_dispatch_args, generate_grouped_synthesis_plan

from deepagent.graph.runtime import (
    MAX_PARALLEL_DISPATCH_FALLBACK,
    _canonical_target_generation_size,
    _desired_generation_size,
    _normalized_parameters,
)
from deepagent.graph.artifacts import _slim_work_item, _write_json_artifact
from deepagent.graph.helpers import _brief_key, _lineage_index, _slot_failure_registry
from deepagent.tracing import (
    _dispatch_fifo_parallel,
    _trace_identity,
)

logger = logging.getLogger(__name__)


def _hydrate_run_working_memory(
    state: AgentState,
    work_items: List[Dict[str, Any]],
    pair_plans: List[Dict[str, Any]],
    selection_strategy: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[int], List[int]]:
    work_items = [dict(item) for item in (work_items or [])]
    pair_plans = [dict(plan) for plan in (pair_plans or [])]
    invariant_bundles = dict(state.get("invariant_bundles", {}) or {})
    pack_items = list(work_items)
    seen_slots = {int(item.get("slot", 0) or 0) for item in pack_items}
    for pair_plan in pair_plans:
        mutation_slot = int(pair_plan.get("mutation_slot", 0) or 0)
        mutation_parent = dict(pair_plan.get("mutation_parent") or {})
        if mutation_slot in seen_slots or not mutation_parent:
            continue
        pack_items.append(
            {
                "slot": mutation_slot,
                "op_type": "mutation",
                "pair_id": pair_plan.get("pair_id"),
                "parent_ids": [mutation_parent.get("id")],
                "parents": [mutation_parent],
                "variation_axis": (
                    pair_plan.get("mutation_variation_axis")
                    or pair_plan.get("variation_axis")
                    or pair_plan.get("bridge_axis")
                    or ""
                ),
                "requires_research": False,
                "invariant_bundles": [bundle for bundle in [pair_plan.get("mutation_invariant_bundle")] if bundle],
                "difficulty_strategy": pair_plan.get("difficulty_strategy", ""),
                "dispatch_rationale": pair_plan.get("rationale", ""),
            }
        )
        seen_slots.add(mutation_slot)

    run_working_memory = build_run_working_memory(
        work_items=pack_items,
        current_generation=state.get("current_generation", []) or [],
        generation_count=state.get("generation_count", 0),
        invariant_bundles=invariant_bundles,
        selection_strategy=selection_strategy,
        failed_problems=state.get("failed_problems", []) or [],
        slot_retry_registry=state.get("slot_retry_registry", {}) or {},
        accepted_candidate_deltas=((state.get("run_working_memory", {}) or {}).get("accepted_candidate_deltas", []) or []),
    )
    context_packs = dict(run_working_memory.get("context_packs", {}) or {})
    pair_plan_map = {plan.get("pair_id"): plan for plan in pair_plans}
    for item in work_items:
        slot_key = str(item.get("slot", 0))
        pack = context_packs.get(slot_key, {})
        item["context_pack"] = pack
        item["context_pack_digest"] = pack.get("digest", "")
        item["requires_research"] = bool(item.get("requires_research", False))
        item["requires_llm_brief"] = bool(should_use_llm_brief(item, pack))
        if item.get("pair_id") in pair_plan_map and item.get("op_type") == "crossover":
            pair_plan_map[item["pair_id"]]["crossover_context_pack"] = pack
            mutation_slot = str(pair_plan_map[item["pair_id"]].get("mutation_slot", 0))
            mutation_pack = context_packs.get(mutation_slot, {})
            if mutation_pack:
                pair_plan_map[item["pair_id"]]["mutation_context_pack"] = mutation_pack

    synthesis_briefs = dict(state.get("synthesis_briefs", {}) or {})
    for item in work_items:
        if item.get("op_type") == "survivor" or item.get("requires_llm_brief"):
            continue
        brief = prepare_synthesis_brief(
            item,
            pair_plan_map.get(item.get("pair_id"), {}),
            item.get("research_artifact") or {},
            brief_role=item.get("op_type", "mutation"),
            context_pack=item.get("context_pack") or {},
            retry_feedback=item.get("validation_feedback") or item.get("constraint_guard_feedback") or "",
            invoke_config={
                "tags": ["deepagent", "briefing", item.get("op_type", "brief"), "deterministic-fast-path"],
                "metadata": {"slot": item.get("slot", 0), "pair_id": item.get("pair_id"), "op_type": item.get("op_type")},
            },
        )
        item["synthesis_brief"] = brief
        synthesis_briefs[_brief_key(item.get("slot", 0))] = brief
        if item.get("op_type") == "crossover" and item.get("pair_id") in pair_plan_map:
            pair_plan_map[item["pair_id"]]["crossover_synthesis_brief"] = brief
        if item.get("op_type") == "mutation" and item.get("pair_id") in pair_plan_map:
            pair_plan_map[item["pair_id"]]["mutation_synthesis_brief"] = brief

    research_skipped_slots = [
        int(item.get("slot", 0) or 0)
        for item in work_items
        if item.get("op_type") != "survivor" and not item.get("requires_research")
    ]
    llm_brief_skipped_slots = [
        int(item.get("slot", 0) or 0)
        for item in work_items
        if item.get("op_type") != "survivor" and not item.get("requires_llm_brief")
    ]
    return (
        work_items,
        list(pair_plan_map.values()),
        run_working_memory,
        synthesis_briefs,
        research_skipped_slots,
        llm_brief_skipped_slots,
    )


def plan_generation_node(state: AgentState):
    parents = state.get("current_generation", []) or []
    params = _normalized_parameters(state)
    params["generation_count"] = state.get("generation_count", 0)
    # Reset per-generation transient retry counters so they don't bleed across
    # generations. Without this, a Gen N that ends via elite backfill (partial
    # save with _validation_retry_count=3) causes Gen N+1 to exhaust its retry
    # budget on the very first validate_candidates call.
    params["_validation_retry_count"] = 0
    params["_inner_repair_loop_count"] = 0
    params["elite_backfill_attempted"] = False
    params["elite_backfill_count"] = 0
    invariant_bundles = dict(state.get("invariant_bundles", {}) or extract_invariant_bundles(parents))
    # Surface prior-generation plan outcomes from archival memory so the selector
    # can favour planned-but-not-realized axes and avoid recurrent failures.
    run_working_memory_for_plan = dict(state.get("run_working_memory", {}) or {})
    prior_plan_outcomes = list(
        ((state.get("archival_memory_handle", {}) or {}).get("plan_outcome_cards", []) or [])
    )
    if prior_plan_outcomes:
        stage_views = dict(run_working_memory_for_plan.get("stage_views", {}) or {})
        selector_view = dict(stage_views.get("selector", {}) or {})
        selector_view["plan_outcome_summary"] = summarize_plan_outcome_cards(prior_plan_outcomes)
        stage_views["selector"] = selector_view
        run_working_memory_for_plan["stage_views"] = stage_views
    plan = plan_generation(
        parents,
        params,
        invariant_bundles=invariant_bundles,
        run_working_memory=run_working_memory_for_plan,
        plan_outcome_cards=prior_plan_outcomes,
        recent_survivor_answers=list(state.get("recent_survivor_answers", []) or []),
    )
    work_items = plan.get("work_items", [])
    pair_plans = plan.get("pair_plans", []) or []
    for item in work_items:
        parent_ids = item.get("parent_ids", []) or []
        item["invariant_bundles"] = [invariant_bundles[pid] for pid in parent_ids if pid in invariant_bundles]
        item["normalization_mode"] = plan.get("strategy", {}).get("normalization_mode", "")
        item["desired_generation_size"] = int(plan.get("desired_generation_size", _desired_generation_size(state, params)))
        item["target_generation_size"] = _canonical_target_generation_size(params)
        item["current_generation_size"] = len(parents)
        item["strategy_source"] = plan.get("strategy", {}).get("strategy_source", "")
    for pair_plan in pair_plans:
        parent_ids = pair_plan.get("parent_ids", []) or []
        pair_plan["invariant_bundles"] = [invariant_bundles[pid] for pid in parent_ids if pid in invariant_bundles]
        mutation_parent_id = pair_plan.get("mutation_parent_id")
        pair_plan["mutation_invariant_bundle"] = invariant_bundles.get(mutation_parent_id)
    hydrated_work_items, hydrated_pair_plans, run_working_memory, synthesis_briefs, research_skipped_slots, llm_brief_skipped_slots = _hydrate_run_working_memory(
        {**state, "invariant_bundles": invariant_bundles},
        work_items,
        pair_plans,
        plan.get("strategy", {}),
    )
    any_research = any(item.get("requires_research") and item.get("op_type") != "survivor" for item in hydrated_work_items)
    any_llm_brief = any(item.get("requires_llm_brief") and item.get("op_type") != "survivor" for item in hydrated_work_items)
    any_non_survivor = any(item.get("op_type") != "survivor" for item in hydrated_work_items)
    # Route to synthesis_plan whenever there is any non-survivor slot; the
    # synthesis plan is the authoritative contract that drives research and the
    # generator, even when research would otherwise be skipped.
    if any_non_survivor:
        next_phase = "synthesis_plan"
    else:
        next_phase = "synthesize_candidates"
    logger.info(f"🧭 Planned {len(work_items)} work items.")
    result = {
        "session_phase": next_phase,
        "parameters": params,
        "selection_strategy": plan.get("strategy", {}),
        "work_items": hydrated_work_items,
        "pair_plans": hydrated_pair_plans,
        "desired_generation_size": int(plan.get("desired_generation_size", _desired_generation_size(state, params))),
        "target_generation_size": _canonical_target_generation_size(params),
        "current_generation_size": len(parents),
        "pair_results": {},
        "pair_health_scores": {},
        "mutation_policies": {},
        "synthesis_briefs": synthesis_briefs,
        "run_working_memory": run_working_memory,
        "context_packs": dict(run_working_memory.get("context_packs", {}) or {}),
        "lineage_metrics": _lineage_index(parents),
        "invariant_bundles": invariant_bundles,
        "research_artifacts": [],
        "approved_candidates": [],
        "retry_registry": {},
        "slot_retry_registry": {},
        "slot_failure_registry": {},
        "failed_problems": [],
        "candidates": [],
        "validation_feedback": [],
        "repair_queue": [],
        "generation_handoff_ready": False,
        "review_action": "",
        "review_payload": {},
        "awaiting_review": False,
        "research_skipped_slots": research_skipped_slots,
        "llm_brief_skipped_slots": llm_brief_skipped_slots,
        "fast_path_slot_count": len(llm_brief_skipped_slots),
        "messages": [SystemMessage(content=f"Planned {len(work_items)} work items.")],
    }
    _write_json_artifact(
        {**state, **result},
        "plan_generation",
        {
            "generation_count": state.get("generation_count", 0),
            "selection_strategy": plan.get("strategy", {}),
            "pair_plans": hydrated_pair_plans,
            "invariant_bundles": invariant_bundles,
            "work_items": hydrated_work_items,
        },
    )
    return result


def synthesis_plan_node(state: AgentState):
    """Op-type-grouped, pre-research synthesis plan stage.

    Groups non-survivor slots by op_type (mutation / crossover) and runs one
    LLM call per group in parallel. Each grouped planner sees ALL slots of its
    type at once and is required to produce portfolio-diverse plans (distinct
    composition patterns, concept hooks, research focuses). At most 2 LLM calls
    are dispatched — one mutation group + one crossover group — compared with the
    previous N-per-slot design.

    The per-slot synthesis plan is attached to each work_item and exposed as
    state ``synthesis_plans`` for downstream nodes.
    """
    work_items = [dict(item) for item in (state.get("work_items", []) or [])]
    generation_count = state.get("generation_count", 0)
    max_parallel = int((state.get("parameters", {}) or {}).get("max_parallel_dispatch", MAX_PARALLEL_DISPATCH_FALLBACK))
    pending = []
    passthrough = []
    for item in work_items:
        if item.get("op_type") == "survivor":
            passthrough.append(item)
        else:
            pending.append(item)

    # Group by op_type for portfolio-coordinated planning.
    mutation_items = [item for item in pending if item.get("op_type") == "mutation"]
    crossover_items = [item for item in pending if item.get("op_type") == "crossover"]

    group_items: List[Dict[str, Any]] = []
    if mutation_items:
        group_items.append({
            "op_type": "mutation",
            "items": mutation_items,
            "_trace_name": "deepagent.synthesis_planner.mutation_group",
        })
    if crossover_items:
        group_items.append({
            "op_type": "crossover",
            "items": crossover_items,
            "_trace_name": "deepagent.synthesis_planner.crossover_group",
        })

    # Build the orchestrator → synthesis_planner dispatch envelope.
    # One tool_call per op_type group; each carries the full list of per-slot
    # dispatch args (pydantic-validated) so the envelope is still slot-level
    # auditable in state.messages.
    dispatch_tool_calls: List[Dict[str, Any]] = []
    for g in group_items:
        slots_args = []
        for item in g["items"]:
            retry_feedback = item.get("validation_feedback") or item.get("constraint_guard_feedback") or ""
            slots_args.append(build_synthesis_plan_dispatch_args(item, retry_feedback=retry_feedback))
        group_dispatch_args = {"op_type": g["op_type"], "slots": slots_args}
        tool_call = build_dispatch_tool_call(f"synthesis_plan.group.{g['op_type']}", group_dispatch_args)
        g["_dispatch_tool_call_id"] = tool_call["id"]
        dispatch_tool_calls.append(tool_call)

    def _group_worker(group_dict: Dict) -> List[Dict]:
        op_type = group_dict["op_type"]
        return generate_grouped_synthesis_plan(
            group_dict["items"],
            op_type=op_type,
            invoke_config={
                "run_name": f"deepagent.synthesis_planner.{op_type}_group.llm",
                "tags": ["deepagent", "synthesis_planner", op_type],
                "metadata": {
                    "generation_count": generation_count,
                    "op_type": op_type,
                    "slot_count": len(group_dict["items"]),
                },
            },
        )

    work_items_by_slot: Dict[int, Dict] = {int(item.get("slot", 0) or 0): item for item in pending}
    planned_items: List[Dict] = list(passthrough)
    synthesis_plans: Dict[str, Dict] = {}
    dispatch_tool_messages: List[Any] = []

    for group_dict, slot_plans in _dispatch_fifo_parallel(group_items, max_parallel, _group_worker):
        dispatch_tool_messages.append(
            wrap_dispatch_result(
                group_dict["_dispatch_tool_call_id"],
                f"synthesis_plan.group.{group_dict['op_type']}",
                {
                    "op_type": group_dict["op_type"],
                    "slot_count": len(slot_plans),
                    "slots": [p.get("slot") for p in slot_plans],
                    "patterns": [p.get("preferred_composition_pattern") for p in slot_plans],
                    "research_required": [p.get("research_required") for p in slot_plans],
                },
            )
        )
        for plan in slot_plans:
            slot = int(plan.get("slot", 0) or 0)
            item = dict(work_items_by_slot.get(slot) or {})
            # Overlay any orchestrator-supplied synthesis_plan_override (currently
            # only ``research_focus`` and ``research_required``) from the
            # regen_planner so the researcher starts from the orchestrator's
            # technique-only focus instead of a stale one re-emitted by the
            # synthesis_planner LLM.
            override = dict(item.get("synthesis_plan_override") or {})
            if override:
                for key, value in override.items():
                    if value not in (None, "", []):
                        plan[key] = value
                # An override with a non-empty research_focus implies
                # research_required == True, unless the override explicitly says
                # otherwise. This matches the regen_planner contract.
                if override.get("research_focus") and "research_required" not in override:
                    plan["research_required"] = True
            merged_item = {**item, "synthesis_plan": plan}
            # Honour the plan's research_required flag. When False the researcher
            # is skipped for this slot regardless of the work item's prior value.
            research_required = bool(plan.get("research_required", True))
            plan_focus = (plan or {}).get("research_focus", "").strip()
            if not research_required:
                merged_item["requires_research"] = False
                merged_item["query_hint"] = ""
                # Drop any stale query_hint from the context pack's research_policy.
                context_pack = dict(merged_item.get("context_pack") or {})
                research_policy = dict(context_pack.get("research_policy") or {})
                if research_policy.get("query_hint"):
                    research_policy["query_hint"] = ""
                    context_pack["research_policy"] = research_policy
                    merged_item["context_pack"] = context_pack
            elif plan_focus:
                merged_item["query_hint"] = plan_focus
                context_pack = dict(merged_item.get("context_pack") or {})
                research_policy = dict(context_pack.get("research_policy") or {})
                research_policy["query_hint"] = plan_focus
                context_pack["research_policy"] = research_policy
                merged_item["context_pack"] = context_pack
            planned_items.append(merged_item)
            synthesis_plans[str(slot)] = plan

    planned_items = sorted(planned_items, key=lambda entry: int(entry.get("slot", 0) or 0))

    any_research = any(
        item.get("requires_research") and item.get("op_type") != "survivor"
        for item in planned_items
    )
    any_llm_brief = any(
        item.get("requires_llm_brief") and item.get("op_type") != "survivor"
        for item in planned_items
    )
    next_phase = "research_candidates" if any_research else (
        "prepare_synthesis_briefs" if any_llm_brief else "synthesize_candidates"
    )
    logger.info(f"🧩 Committed {len(synthesis_plans)} synthesis plans ({len(group_items)} groups).")
    dispatch_messages: List[Any] = []
    if dispatch_tool_calls:
        dispatch_messages.append(
            build_dispatch_envelope(
                dispatch_tool_calls,
                content=f"Dispatching {len(group_items)} op-type groups to synthesis_planner.",
            )
        )
        dispatch_messages.extend(dispatch_tool_messages)
    dispatch_messages.append(SystemMessage(content=f"Committed {len(synthesis_plans)} synthesis plans."))
    result = {
        "session_phase": next_phase,
        "work_items": planned_items,
        # Stable snapshot of the planned work_items for plan_outcome_cards.
        # Survives regenerate_failed_node which clears work_items during retry
        # loops; reset at next generation's init_run_memory.
        "planned_work_items": [dict(item) for item in planned_items],
        "synthesis_plans": synthesis_plans,
        "messages": dispatch_messages,
    }
    _write_json_artifact(
        {**state, **result},
        "synthesis_plan",
        {
            "generation_count": generation_count,
            "synthesis_plans": synthesis_plans,
        },
    )
    return result


def research_candidates_node(state: AgentState):
    work_items = state.get("work_items", []) or []
    generation_count = state.get("generation_count", 0)
    researched_items = []
    artifacts = []
    max_parallel = int((state.get("parameters", {}) or {}).get("max_parallel_dispatch", MAX_PARALLEL_DISPATCH_FALLBACK))
    pending = []
    for item in work_items:
        if item.get("op_type") == "survivor":
            researched_items.append({**item, "research_artifact": None})
        elif not should_use_research(item):
            artifact = {
                "query": "",
                "tool_used": "none",
                "sources": [],
                "short_synthesis": "Family-local deterministic strategy is preferred; external research skipped.",
                "degraded": False,
                "degraded_reason": "",
                "conflict_note": "",
            }
            researched_items.append({**item, "research_artifact": artifact, "requires_research": False})
            artifacts.append({"slot": item.get("slot"), "pair_id": item.get("pair_id"), **artifact})
        else:
            item["_trace_name"] = _trace_identity(item, "deepagent.researcher")
            pending.append(item)

    # Build the orchestrator → researcher dispatch envelope per slot.
    dispatch_tool_calls: List[Dict[str, Any]] = []
    slot_to_tool_call_id: Dict[int, str] = {}
    for item in pending:
        args_payload = build_research_dispatch_args(item)
        tool_call = build_dispatch_tool_call("researcher.dispatch", args_payload)
        item["_dispatch_args"] = args_payload
        item["_dispatch_tool_call_id"] = tool_call["id"]
        slot_to_tool_call_id[int(item.get("slot", 0) or 0)] = tool_call["id"]
        dispatch_tool_calls.append(tool_call)

    def _research_worker(item: Dict):
        item_identity = item.get("pair_id") or f"slot_{item.get('slot', 0)}"
        tool_call_id = item.get("_dispatch_tool_call_id", "")
        invoke_config = {
            "run_name": f"deepagent.researcher.{item_identity}",
            "tags": ["deepagent", "researcher"],
            "metadata": {
                "generation_count": generation_count,
                "slot": item.get("slot"),
                "op_type": item.get("op_type"),
                "pair_id": item.get("pair_id"),
                "dispatch_tool_call_id": tool_call_id,
            },
        }
        return research_work_item(
            item,
            invoke_config=invoke_config,
            dispatch_args=item.get("_dispatch_args"),
        )

    dispatch_tool_messages: List[Any] = []
    for item, artifact in _dispatch_fifo_parallel(pending, max_parallel, _research_worker):
        slot = int(item.get("slot", 0) or 0)
        tool_call_id = item.get("_dispatch_tool_call_id") or slot_to_tool_call_id.get(slot, "")
        # Strip internal bookkeeping before persisting the item into state.
        clean_item = {k: v for k, v in item.items() if k not in {"_dispatch_args", "_dispatch_tool_call_id"}}
        researched_items.append({**clean_item, "research_artifact": artifact})
        artifacts.append({"slot": item.get("slot"), "pair_id": item.get("pair_id"), **artifact})
        dispatch_tool_messages.append(
            wrap_dispatch_result(
                tool_call_id,
                "researcher.dispatch",
                {
                    "slot": slot,
                    "pair_id": item.get("pair_id", ""),
                    "query": artifact.get("query", ""),
                    "tool_used": artifact.get("tool_used", ""),
                    "source_count": len(artifact.get("sources", []) or []),
                    "degraded": artifact.get("degraded", False),
                },
            )
        )
    any_llm_brief = any(
        item.get("op_type") != "survivor" and should_use_llm_brief(item, item.get("context_pack") or {})
        for item in researched_items
    )
    dispatch_messages: List[Any] = []
    if dispatch_tool_calls:
        dispatch_messages.append(
            build_dispatch_envelope(
                dispatch_tool_calls,
                content=f"Dispatching {len(dispatch_tool_calls)} slots to researcher.",
            )
        )
        dispatch_messages.extend(dispatch_tool_messages)
    dispatch_messages.append(SystemMessage(content=f"Researched {len(artifacts)} work items."))
    result = {
        "session_phase": "prepare_synthesis_briefs" if any_llm_brief else "synthesize_candidates",
        "work_items": sorted(researched_items, key=lambda item: item.get("slot", 0)),
        "research_artifacts": artifacts,
        "research_skipped_slots": [
            int(item.get("slot", 0) or 0)
            for item in researched_items
            if item.get("op_type") != "survivor" and (item.get("research_artifact") or {}).get("tool_used") == "none"
        ],
        "messages": dispatch_messages,
    }
    _write_json_artifact(
        {**state, **result},
        "research_candidates",
        {
            "generation_count": state.get("generation_count", 0),
            "research_artifacts": artifacts,
            "research_skipped_slots": result.get("research_skipped_slots", []),
        },
    )
    return result


def prepare_synthesis_briefs_node(state: AgentState):
    """Brief validator — merges the pre-research synthesis plan with the research
    artifact into the final generator-facing brief.

    Runs per-slot in parallel; each slot's work is independent because the
    synthesis plan has already committed the research-independent fields.
    """
    work_items = [dict(item) for item in (state.get("work_items", []) or [])]
    pair_plans = [dict(plan) for plan in (state.get("pair_plans", []) or [])]
    synthesis_briefs = dict(state.get("synthesis_briefs", {}) or {})
    synthesis_plans = dict(state.get("synthesis_plans", {}) or {})
    pair_plan_map = {plan.get("pair_id"): plan for plan in pair_plans}
    max_parallel = int((state.get("parameters", {}) or {}).get("max_parallel_dispatch", MAX_PARALLEL_DISPATCH_FALLBACK))

    pending_primary: List[Dict] = []
    for item in work_items:
        if item.get("op_type") == "survivor":
            continue
        pending_primary.append(item)

    def _brief_worker(item: Dict):
        item_identity = item.get("pair_id") or f"slot_{item.get('slot', 0)}"
        retry_feedback = item.get("validation_feedback") or item.get("constraint_guard_feedback") or ""
        synthesis_plan = synthesis_plans.get(str(item.get("slot", 0))) or item.get("synthesis_plan") or {}
        brief = prepare_synthesis_brief(
            item,
            pair_plan_map.get(item.get("pair_id"), {}),
            item.get("research_artifact") or {},
            brief_role=item.get("op_type", "mutation"),
            context_pack=item.get("context_pack") or {},
            retry_feedback=retry_feedback,
            synthesis_plan=synthesis_plan,
            invoke_config={
                "run_name": f"deepagent.brief_validator.{item_identity}.llm",
                "tags": ["deepagent", "brief_validator", item.get("op_type", "brief")],
                "metadata": {"slot": item.get("slot", 0), "pair_id": item.get("pair_id"), "op_type": item.get("op_type")},
            },
        )
        return brief

    work_items_by_slot = {int(entry.get("slot", 0) or 0): entry for entry in work_items}
    for item, brief in _dispatch_fifo_parallel(pending_primary, max_parallel, _brief_worker):
        slot = int(item.get("slot", 0) or 0)
        target = work_items_by_slot.get(slot, item)
        target["synthesis_brief"] = brief
        synthesis_briefs[_brief_key(slot)] = brief
        if target.get("op_type") == "crossover" and target.get("pair_id") in pair_plan_map:
            pair_plan_map[target["pair_id"]]["crossover_synthesis_brief"] = brief

    # Crossover-paired mutations: build in a second parallel wave using the
    # mutation parent's own synthesis plan where available, falling back to the
    # pair plan's mutation context pack.
    pending_secondary: List[Dict] = []
    for target in work_items:
        if target.get("op_type") != "crossover":
            continue
        pair_plan = pair_plan_map.get(target.get("pair_id"))
        if not pair_plan:
            continue
        mutation_slot = pair_plan.get("mutation_slot", 0)
        if _brief_key(mutation_slot) in synthesis_briefs:
            continue
        pending_secondary.append(
            {
                "slot": mutation_slot,
                "pair_id": target.get("pair_id"),
                "op_type": "mutation",
                "variation_axis": (
                    pair_plan.get("mutation_variation_axis")
                    or pair_plan.get("variation_axis")
                    or pair_plan.get("bridge_axis")
                    or ""
                ),
                "bridge_axis": pair_plan.get("bridge_axis", ""),
                "research_artifact": target.get("research_artifact") or {},
                "context_pack": pair_plan.get("mutation_context_pack") or {},
                "validation_feedback": target.get("validation_feedback", ""),
                "constraint_guard_feedback": target.get("constraint_guard_feedback", ""),
                "_pair_id": target.get("pair_id"),
            }
        )

    def _mutation_brief_worker(item: Dict):
        item_identity = f"{item.get('pair_id')}_mutation"
        retry_feedback = item.get("validation_feedback") or item.get("constraint_guard_feedback") or ""
        synthesis_plan = synthesis_plans.get(str(item.get("slot", 0))) or {}
        brief = prepare_synthesis_brief(
            item,
            pair_plan_map.get(item.get("_pair_id"), {}),
            item.get("research_artifact") or {},
            brief_role="mutation",
            context_pack=item.get("context_pack") or {},
            retry_feedback=retry_feedback,
            synthesis_plan=synthesis_plan,
            invoke_config={
                "run_name": f"deepagent.brief_validator.{item_identity}.llm",
                "tags": ["deepagent", "brief_validator", "mutation"],
                "metadata": {"slot": item.get("slot", 0), "pair_id": item.get("_pair_id"), "op_type": "mutation"},
            },
        )
        return brief

    for item, brief in _dispatch_fifo_parallel(pending_secondary, max_parallel, _mutation_brief_worker):
        slot = int(item.get("slot", 0) or 0)
        pair_id = item.get("_pair_id")
        if pair_id in pair_plan_map:
            pair_plan_map[pair_id]["mutation_synthesis_brief"] = brief
        synthesis_briefs[_brief_key(slot)] = brief

    updated_pair_plans = list(pair_plan_map.values())
    result = {
        "session_phase": "synthesize_candidates",
        "work_items": work_items,
        "pair_plans": updated_pair_plans,
        "synthesis_briefs": synthesis_briefs,
        "llm_brief_skipped_slots": list(state.get("llm_brief_skipped_slots", []) or []),
        "fast_path_slot_count": int(state.get("fast_path_slot_count", 0) or 0),
        "messages": [SystemMessage(content=f"Prepared {len(synthesis_briefs)} synthesis briefs.")],
    }
    _write_json_artifact(
        {**state, **result},
        "prepare_synthesis_briefs",
        {
            "generation_count": state.get("generation_count", 0),
            "synthesis_briefs": synthesis_briefs,
        },
    )
    return result
