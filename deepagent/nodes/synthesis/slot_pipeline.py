"""Option A — slot fan-out via LangGraph Send API.

Replaces the synthesize_candidates → postprocess_candidates phase barrier with
a per-slot pipeline that runs synthesize + contract repair + postprocess
independently for each slot. Each slot becomes a parallel Send branch; an
aggregator joins results and routes to the existing validate_candidates node.

Contract:
- `slot_dispatch_node(state)` is a regular state-mutating node. It splits
  work_items into survivors (immediately added to `slot_outputs` for the
  current generation) and non-survivors (emitted via `slot_dispatch_route`
  as `Send("slot_unit", payload)` instances).
- `slot_unit_node(payload)` runs ONE slot's synthesize + (contract repair) +
  postprocess. It returns `{"slot_outputs": [single_dict]}`; the
  `operator.add` reducer on `slot_outputs` accumulates entries from all
  parallel branches without races.
- `slot_aggregate_node(state)` filters slot_outputs by current
  `generation_count`, projects them into the existing `candidates` /
  `failed_problems` shape, attaches peer-context, builds dispatch
  tool-messages, and routes to validate_candidates.

Per-generation slot_outputs entries from PRIOR generations remain in state
(small constant memory cost) but are filtered out by the aggregator on each
new generation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage
from langgraph.types import Send
from langsmith.run_helpers import get_current_run_tree, tracing_context

from deepagent.graph.artifacts import _write_json_artifact
from deepagent.graph.helpers import (
    _apply_regen_decision_to_retry_item,
    _build_slot_failure_summary,
    _clear_nonblocking_failed_when_population_full,
    _contract_assessment,
    _derive_child_lineage_metrics,
    _ensure_unique_problem_id,
    _failure_type_from_reason,
    _lineage_metrics_for_problem,
    _make_failed_problem,
    _prune_failed_for_approved_slots,
    _reason_signature,
    _record_slot_failure,
    _repair_strategy_for_failure,
    _select_elite_backfill,
    _slot_failure_registry,
)
from deepagent.graph.runtime import _desired_generation_size
from deepagent.nodes.synthesis.generator import (
    deep_crossover_problems,
    deep_mutate_problem,
)
from deepagent.nodes.synthesis.orchestrator import (
    _attach_peer_context,
    _build_synthesis_dispatch_args,
    _run_repair_item,
)
# Worker functions live in validation.worker (leaf module); importing from
# validation.orchestrator triggers validation/__init__ → graph package
# circular import. Use the worker module directly.
from deepagent.nodes.validation.worker import (
    build_execution_evidence,
    ground_solution,
    materialize_answer_from_code,
)
# Phase B (Option B1): validate per-slot. validate_one_candidate IS in
# validation.orchestrator but it's a pure function — lazy-import inside
# slot_unit_node to avoid the circular-import issue (orchestrator → graph).
from deepagent.state_full import AgentState
from deepagent.tracing import (
    build_dispatch_envelope,
    build_dispatch_tool_call,
    close_one_slot_trace,
    ensure_slot_trace_parent,
    wrap_dispatch_result,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Pure helpers — synthesis and postprocess for ONE slot
# ─────────────────────────────────────────────────────────────────────────


def _run_one_synthesis(item: Dict[str, Any], generation_count: int) -> Dict[str, Any]:
    """Mirror of `_synthesis_worker` from the legacy node, freed of closure
    state. Returns {'candidate': child, 'error': ''} or {'candidate': None,
    'error': str}.
    """
    try:
        slot = item.get("slot", 0)
        op_type = item.get("op_type")
        parents = item.get("parents", [])
        item_identity = item.get("pair_id") or f"slot_{slot}"
        invoke_config = {
            "run_name": f"deepagent.generator.{op_type}.{item_identity}",
            "tags": ["deepagent", "generator", op_type],
            "metadata": {
                "slot": slot,
                "pair_id": item.get("pair_id"),
                "op_type": op_type,
                "normalization_mode": item.get("normalization_mode", ""),
                "desired_generation_size": item.get("desired_generation_size"),
                "target_generation_size": item.get("target_generation_size"),
                "current_generation_size": item.get("current_generation_size"),
                "strategy_source": item.get("strategy_source", ""),
            },
        }
        if op_type == "mutation" and parents:
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
                validation_feedback=item.get("validation_feedback", ""),
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
                validation_feedback=item.get("validation_feedback", ""),
            )
        else:
            raise ValueError(f"Unsupported work item: {op_type}")
        child["_slot"] = slot
        child["pair_id"] = item.get("pair_id")
        child["op_type"] = op_type
        child["generation_meta"] = {
            "generation_count": int(generation_count),
            "slot": slot,
            "op_type": op_type,
            "pair_id": item.get("pair_id"),
            "context_pack_digest": item.get("context_pack_digest", ""),
        }
        child["lineage_metrics"] = _derive_child_lineage_metrics(parents, op_type)
        return {"candidate": child, "error": ""}
    except Exception as exc:  # noqa: BLE001 — preserve legacy contract
        return {"candidate": None, "error": str(exc)}


def _run_one_postprocess(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror of `_postprocess_worker` from validation orchestrator. Returns
    {'ok': True, 'candidate': updated} or {'ok': False, 'failure_reason': ...}.
    """
    slot = candidate.get("_slot", candidate.get("slot", 0))
    ok, canonical_answer, reason = materialize_answer_from_code(candidate)
    if not ok:
        return {"ok": False, "failure_reason": f"Answer materialization failed: {reason}", "slot": slot}
    candidate["worker_answer"] = str(candidate.get("answer", ""))
    candidate["answer"] = canonical_answer
    candidate["orchestrator_materialized_answer"] = canonical_answer
    candidate["answer_materialization_reason"] = reason
    execution_evidence = build_execution_evidence(candidate)
    grounding = ground_solution(
        candidate,
        invoke_config={
            "run_name": f"deepagent.grounding.slot_{slot}",
            "tags": ["deepagent", "grounding"],
            "metadata": {
                "slot": slot,
                "pair_id": candidate.get("pair_id"),
                "problem_id": candidate.get("id", ""),
            },
        },
    )
    candidate["execution_evidence"] = execution_evidence
    candidate["grounding_assessment"] = grounding
    candidate["grounded_solution_version"] = grounding.get("grounded_solution", "")
    if grounding.get("grounding_status") == "grounded" and grounding.get("grounded_solution"):
        candidate["solution"] = grounding.get("grounded_solution", candidate.get("solution", ""))
        candidate["evidence_summary"] = grounding.get("grounded_evidence_summary", candidate.get("evidence_summary", ""))
    return {"ok": True, "candidate": candidate}


# ─────────────────────────────────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────────────────────────────────


def slot_dispatch_node(state: AgentState) -> Dict[str, Any]:
    """Prepare survivors and a per-slot dispatch envelope; return state
    updates only. The actual fan-out happens via `slot_dispatch_route`
    (the conditional-edge function that returns the Send list).
    """
    work_items = list(state.get("work_items", []) or [])
    generation_count = int(state.get("generation_count", 0))
    survivors_outputs: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for item in work_items:
        slot = item.get("slot", 0)
        op_type = item.get("op_type")
        parents = item.get("parents", [])
        if op_type == "survivor":
            if parents:
                survivor = dict(parents[0])
                survivor["_slot"] = slot
                survivor["type"] = "survivor"
                survivor["op_type"] = "survivor"
                survivor["lineage_metrics"] = _lineage_metrics_for_problem(survivor)
                survivors_outputs.append(
                    _wrap_slot_output(generation_count, slot, candidate=survivor, kind="survivor")
                )
            continue
        pending.append(item)

    # Build the orchestrator → generator dispatch envelope (mirrors the legacy
    # synthesize_candidates_node) so the LangSmith trace still records the
    # dispatch tool_call semantics. We attach the resulting tool_call_id map
    # to state for the aggregator to wrap each slot's outcome.
    generator_tool_calls: List[Dict[str, Any]] = []
    slot_to_generator_tool_call_id: Dict[str, str] = {}
    for item in pending:
        try:
            args_payload = _build_synthesis_dispatch_args(item)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"generator dispatch args projection failed for slot {item.get('slot')}: {exc}")
            continue
        tool_call = build_dispatch_tool_call("generator.dispatch", args_payload)
        slot_to_generator_tool_call_id[str(int(item.get("slot", 0) or 0))] = tool_call["id"]
        generator_tool_calls.append(tool_call)

    dispatch_messages: List[Any] = []
    if generator_tool_calls:
        dispatch_messages.append(
            build_dispatch_envelope(
                generator_tool_calls,
                content=f"Dispatching {len(generator_tool_calls)} slots to generator (fan-out).",
            )
        )

    return {
        "session_phase": "slot_unit",
        "slot_outputs": survivors_outputs,  # operator.add — append survivors
        # Stash the tool_call ID map so aggregator can build matching ToolMessages
        "run_metadata": {
            **(state.get("run_metadata", {}) or {}),
            "slot_dispatch_tool_call_ids": slot_to_generator_tool_call_id,
            "slot_dispatch_pending_count": len(pending),
        },
        "messages": dispatch_messages,
    }


def slot_dispatch_route(state: AgentState):
    """Conditional-edge function: emit one `Send("slot_unit", payload)` per
    non-survivor work_item. If there are no non-survivors, route directly to
    the aggregator (which handles the survivor-only case).
    """
    work_items = list(state.get("work_items", []) or [])
    generation_count = int(state.get("generation_count", 0))
    session_thread_id = str(state.get("session_thread_id", "") or "")
    # Snapshot registry so repair workers can surface prior_attempts history
    # (Phase 3 / R-2). Send branches each get a copy.
    slot_failure_registry = dict(_slot_failure_registry(state))
    # Phase B (validate per-slot): slot_unit needs subset of state to call
    # validate_one_candidate. Pass full invariant_bundles dict (typically
    # small — keyed by parent_id) and archival_memory_handle.
    invariant_bundles_all = dict(state.get("invariant_bundles", {}) or {})
    archival_memory_handle = dict(state.get("archival_memory_handle", {}) or {})
    parents_by_id = {
        p.get("id"): p
        for p in (state.get("current_generation", []) or [])
        if p.get("id")
    }
    sends: List[Send] = []
    for item in work_items:
        op_type = item.get("op_type")
        if op_type == "survivor":
            continue
        # Filter invariant_bundles + parents to just this slot's parent_ids,
        # keeping the per-Send payload size bounded.
        slot_parent_ids = list(item.get("parent_ids", []) or [])
        invariant_bundles_for_slot = {
            pid: invariant_bundles_all[pid]
            for pid in slot_parent_ids
            if pid in invariant_bundles_all
        }
        parents_for_slot = {
            pid: parents_by_id[pid]
            for pid in slot_parent_ids
            if pid in parents_by_id
        }
        payload = {
            "work_item": item,
            "generation_count": generation_count,
            "session_thread_id": session_thread_id,
            "slot_failure_registry": slot_failure_registry,
            "invariant_bundles_for_slot": invariant_bundles_for_slot,
            "parents_for_slot": parents_for_slot,
            "archival_memory_handle": archival_memory_handle,
            # Phase F: slot_unit's per-slot retry loop reads max_slot_regen_attempts
            # and max_research_refetch from here.
            "parameters": dict(state.get("parameters", {}) or {}),
        }
        sends.append(Send("slot_unit", payload))
    if not sends:
        return "slot_aggregate"
    return sends


def slot_unit_node(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run synthesize + (contract repair) + postprocess for ONE slot.
    Returns a single-entry slot_outputs delta which the operator.add reducer
    folds into the global list.

    Note: payload IS this node's "state" (LangGraph Send semantics). The
    parent state's other fields are NOT visible here.
    """
    item = dict(payload.get("work_item", {}) or {})
    generation_count = int(payload.get("generation_count", 0) or 0)
    slot = int(item.get("slot", 0) or 0)
    op_type = item.get("op_type", "")

    # Tag for per-slot trace parent registration. The trace registry uses
    # (session_thread_id, generation, slot) as key so concurrent Send branches
    # do not collide.
    slot_trace_state = {
        "session_thread_id": payload.get("session_thread_id", ""),
        "generation_count": generation_count,
    }
    item["_slot_trace_enabled"] = True
    item["_slot_trace_name"] = f"generator.operation.slot_{slot}"
    item["_slot_trace_tags"] = ["deepagent", "generator", "operation", str(op_type or "worker")]
    # Phase D.3: enrich slot trace metadata for LangSmith UI filtering /
    # identification. `pipeline_phase` distinguishes initial synth from
    # repair retry; `parent_ids` exposes the lineage; `mode` /
    # `difficulty_label` make easy/hard filtering possible without diving
    # into individual LLM call metadata.
    incoming_repair_strategy = str(item.get("repair_strategy") or "").strip()
    pipeline_phase = "repair_retry" if (incoming_repair_strategy and incoming_repair_strategy != "_giveup") else "initial_synth"
    item["_slot_trace_metadata"] = {
        "slot": slot,
        "pair_id": item.get("pair_id"),
        "op_type": op_type,
        "variation_axis": item.get("variation_axis", ""),
        "pipeline_phase": pipeline_phase,
        "mode": item.get("mode", ""),
        "difficulty_label": item.get("difficulty_label", ""),
        "parent_ids": list(item.get("parent_ids", []) or []),
        "repair_strategy": incoming_repair_strategy,
        "generation_count": generation_count,
    }
    parent_run = get_current_run_tree()
    slot_run = ensure_slot_trace_parent(slot_trace_state, item, parent_run=parent_run)

    # Phase D.1 + Phase E (final): enter the slot trace parent's context so
    # ALL nested LLM / tool spans (deep_mutate's `.plan.llm`, `.structured.llm`,
    # `.explore`, postprocess `grounding.slot_*.llm`, validate_one_candidate's
    # `solvability.*`, `validator.*`, `quality.*` spans, AND per-slot
    # regen_plan calls made from regenerate_failed_node via
    # ensure_slot_trace_parent + tracing_context lookup) nest under
    # `generator.operation.slot_{slot}` for the full slot lifecycle.
    #
    # Phase E: do NOT close the slot trace span here — keep it open across
    # all retry attempts so the slot's full lifecycle (synth → validate →
    # per-slot regen_plan → hotfix → validate → ...) stays under one span.
    # `close_slot_trace_parents` (called by save_generation) closes all
    # slot spans for the generation at once.
    with tracing_context(parent=slot_run if slot_run is not None else parent_run):
        return _slot_unit_body(item, payload, generation_count, slot, op_type)


def _do_one_attempt(
    current_item: Dict[str, Any],
    payload: Dict[str, Any],
    generation_count: int,
    slot: int,
    candidate_before_repair: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Execute ONE attempt: synth-or-repair → contract repair → postprocess →
    validate. Returns a dict describing the outcome:
        {"status": "passed", "candidate": <dict>, "contract_repair_strategy": str}
        {"status": "failed", "stage": "synthesis"|"contract_repair"|"postprocess"|"validate",
         "failure_type": str, "failure_reason": str, "candidate": <dict-or-None>}
    """
    incoming_repair_strategy = str(current_item.get("repair_strategy") or "").strip()
    is_repair_retry = bool(incoming_repair_strategy) and incoming_repair_strategy != "_giveup"

    # Step 1: synth or repair
    if is_repair_retry:
        # Hotfix paths require candidate_before_repair; full_regenerate doesn't.
        if incoming_repair_strategy in {"code_only_hotfix", "target_only_hotfix", "statement_domain_hotfix"}:
            if candidate_before_repair:
                current_item["candidate_before_repair"] = candidate_before_repair
        current_item["_trace_name"] = f"orchestrator.repair_failed_candidates.slot_{slot}"
        registry_snapshot = dict(payload.get("slot_failure_registry", {}) or {})
        outcome = _run_repair_item(
            {"generation_count": generation_count},
            current_item,
            registry_snapshot,
        )
        if outcome.get("error"):
            return {
                "status": "failed", "stage": "synthesis",
                "failure_type": current_item.get("failure_type", "regenerability_failure"),
                "failure_reason": f"Repair failed: {outcome['error']}",
                "candidate": None,
                "incoming_repair_strategy": incoming_repair_strategy,
            }
    else:
        outcome = _run_one_synthesis(current_item, generation_count)
        if outcome.get("error"):
            return {
                "status": "failed", "stage": "synthesis",
                "failure_type": "synthesis_error",
                "failure_reason": f"Synthesis failed: {outcome['error']}",
                "candidate": None,
            }
    candidate = outcome["candidate"]

    # Step 2: contract assessment + in-iteration repair (self-heal once
    # per attempt before postprocess)
    contract_repair_strategy = ""
    assessment = _contract_assessment(candidate)
    if assessment.get("repairable"):
        contract_repair_strategy = _repair_strategy_for_failure(
            assessment["failure_type"],
            _reason_signature(assessment["failure_reason"] or assessment["failure_type"]),
        )
        repair_item = dict(current_item)
        slim_candidate = {
            k: candidate.get(k)
            for k in (
                "id", "_slot", "slot", "op_type", "parent_ids",
                "statement", "answer", "solution", "code",
                "code_runtime_mode", "evidence_summary",
                "variation_axis", "variation_axis_used",
            )
            if candidate.get(k) is not None
        }
        repair_item.update({
            "slot": slot,
            "repair_strategy": contract_repair_strategy,
            "validation_feedback": assessment["failure_reason"] or assessment["failure_type"],
            "failure_type": assessment["failure_type"],
            "failure_stage": assessment["failure_stage"],
            "candidate_before_repair": slim_candidate,
            "_trace_name": f"orchestrator.repair_failed_candidates.slot_{slot}",
        })
        registry_snapshot: Dict[str, Any] = {}
        repair_outcome = _run_repair_item({"generation_count": generation_count}, repair_item, registry_snapshot)
        if repair_outcome.get("error"):
            return {
                "status": "failed", "stage": "contract_repair",
                "failure_type": assessment["failure_type"],
                "failure_reason": f"Contract repair failed: {repair_outcome['error']}",
                "candidate": candidate,
                "contract_repair_strategy": contract_repair_strategy,
            }
        candidate = repair_outcome["candidate"]

    # Step 3: postprocess
    pp = _run_one_postprocess(candidate)
    if not pp["ok"]:
        return {
            "status": "failed", "stage": "postprocess",
            "failure_type": "code_gate_failure",
            "failure_reason": pp["failure_reason"],
            "candidate": candidate,
            "contract_repair_strategy": contract_repair_strategy,
        }

    # Step 4: per-slot validate
    from deepagent.nodes.validation.orchestrator import validate_one_candidate
    validated = pp["candidate"]
    try:
        passed, reason = validate_one_candidate(
            validated,
            invariant_bundles_for_parents=dict(payload.get("invariant_bundles_for_slot", {}) or {}),
            parents_for_candidate=dict(payload.get("parents_for_slot", {}) or {}),
            archival_memory_handle=dict(payload.get("archival_memory_handle", {}) or {}),
            generation_count=generation_count,
            ablation_condition=str((payload.get("parameters", {}) or {}).get("ablation_condition", "full")),
        )
    except Exception as exc:  # noqa: BLE001
        passed = False
        reason = f"Validate raised: {type(exc).__name__}: {exc}"
        logger.warning("[slot_unit] slot %s validate raised %s", slot, exc)

    if not passed:
        return {
            "status": "failed", "stage": "validate",
            "failure_type": _failure_type_from_reason(reason),
            "failure_reason": reason,
            "candidate": validated,
            "contract_repair_strategy": contract_repair_strategy,
        }

    # Step 5: per-slot ground_and_rescore (Phase G).
    # Slots that pass validate STILL need to clear the grounding gate
    # before being approved. Running it here (inside slot_unit, under the
    # slot's tracing context) means:
    #   - the `ground_and_rescore.slot_N` LLM nests under
    #     `generator.operation.slot_N` for clean per-slot lifecycle traces
    #   - slots that pass both validate AND ground exit IMMEDIATELY (no
    #     waiting for a cross-slot ground_and_rescore_node aggregator)
    #   - ground rescope/reject feeds back into THIS slot's regen loop
    #     just like a validate failure (per-slot retry consistency)
    from deepagent.nodes.validation.orchestrator import run_one_ground_and_rescore
    try:
        ground_result = run_one_ground_and_rescore(
            validated,
            parents_for_candidate=dict(payload.get("parents_for_slot", {}) or {}),
            synthesis_brief=current_item.get("synthesis_brief") or payload.get("synthesis_brief_for_slot") or {},
            sandbox_evidence=validated.get("validation_evidence") or {},
            archival_memory_handle=dict(payload.get("archival_memory_handle", {}) or {}),
            op_type=str(validated.get("op_type") or current_item.get("op_type", "") or "").strip(),
        )
    except Exception as exc:  # noqa: BLE001 — ground exception treated as soft failure
        logger.warning("[slot_unit] slot %s ground raised %s", slot, exc)
        return {
            "status": "failed", "stage": "ground",
            "failure_type": "ground_error",
            "failure_reason": f"Ground raised: {type(exc).__name__}: {exc}",
            "candidate": validated,
            "contract_repair_strategy": contract_repair_strategy,
        }

    ground_decision = ground_result.get("decision", "accept")
    if ground_decision == "accept":
        return {
            "status": "passed", "candidate": validated,
            "contract_repair_strategy": contract_repair_strategy,
        }
    # rescope / reject — soft failure, retry path
    ground_reason = ground_result.get("reason") or f"Grounding gate {ground_decision}"
    return {
        "status": "failed", "stage": "ground",
        "failure_type": "ground_rescope" if ground_decision == "rescope" else "ground_reject",
        "failure_reason": f"[{ground_decision}] {ground_reason}",
        "candidate": validated,
        "contract_repair_strategy": contract_repair_strategy,
    }


def _slot_unit_body(
    item: Dict[str, Any],
    payload: Dict[str, Any],
    generation_count: int,
    slot: int,
    op_type: str,
) -> Dict[str, Any]:
    """Per-slot retry loop (Phase F).

    Each Send branch runs its OWN multi-attempt retry loop independently.
    No fan-in barrier between attempts: as soon as slot N's attempt finishes
    and validate fails, this branch IMMEDIATELY calls per-slot regen_plan
    (the LLM call nests under this slot's `generator.operation.slot_N`
    span) and dispatches the next attempt — without waiting for other
    slots' progress.

    Loop stops when:
      (a) An attempt passes validate → returns `validated_pass`
      (b) regen_plan returns `giveup` → returns `validated_fail` with reason
      (c) max_slot_regen_attempts reached → returns `validated_fail`
    """
    # Lazy import to break synthesis → regen → tracing import cycle.
    from deepagent.nodes.regen.regen_planner import plan_one_slot_regen

    parameters = payload.get("parameters", {}) or {}
    max_attempts = int(parameters.get("max_slot_regen_attempts", 4))
    research_refetch_budget = int(parameters.get("max_research_refetch", 1))

    current_item = dict(item)
    candidate_before: Dict[str, Any] = None  # last produced candidate (for hotfix)
    history_failure_types: List[str] = []
    history_signatures: List[str] = []
    attempted_strategies: List[str] = []
    research_refetch_count = 0
    last_failure_reason = ""
    last_failure_type = ""

    for attempt_num in range(1, max_attempts + 1):
        # Record the strategy that this attempt will actually execute. A1 starts
        # fresh (no repair_strategy set on current_item) → "synth"; later attempts
        # carry the repair_strategy set by _apply_regen_decision_to_retry_item.
        current_strategy = str(current_item.get("repair_strategy") or "").strip() or "synth"
        attempted_strategies.append(current_strategy)

        result = _do_one_attempt(
            current_item, payload, generation_count, slot,
            candidate_before_repair=candidate_before,
        )

        if result["status"] == "passed":
            return {
                "slot_outputs": [
                    _wrap_slot_output(
                        generation_count, slot,
                        candidate=result["candidate"],
                        kind="validated_pass",
                        contract_repair_strategy=result.get("contract_repair_strategy", ""),
                        validation_passed=True,
                        validation_reason="",
                    )
                ]
            }

        # Failed — accumulate history, decide whether to retry
        last_failure_type = result.get("failure_type", "regenerability_failure")
        last_failure_reason = result.get("failure_reason", "")
        history_failure_types.append(last_failure_type)
        history_signatures.append(_reason_signature(last_failure_reason))
        if result.get("candidate") is not None:
            candidate_before = {
                k: result["candidate"].get(k)
                for k in (
                    "id", "_slot", "slot", "op_type", "parent_ids",
                    "statement", "answer", "solution", "code",
                    "code_runtime_mode", "evidence_summary",
                    "variation_axis", "variation_axis_used",
                )
                if result["candidate"].get(k) is not None
            }

        if attempt_num >= max_attempts:
            break

        # Build slot summary for per-slot regen LLM
        synthetic_failed = dict(current_item)
        synthetic_failed.update({
            "_slot": slot, "slot": slot,
            "op_type": current_item.get("op_type", "mutation"),
            "validation_feedback": last_failure_reason,
            "failure_type": last_failure_type,
            "failure_stage": result.get("stage", "validate"),
            "failure_signature": _reason_signature(last_failure_reason),
        })
        registry_entry = {
            "attempts": attempt_num,
            "failure_types": history_failure_types[-6:],
            "failure_stages": [result.get("stage", "validate")] * min(attempt_num, 6),
            "reason_signatures": history_signatures[-6:],
            "research_refetch_count": research_refetch_count,
            "repair_strategies": attempted_strategies[-6:],
        }
        repeated = (len(history_signatures) >= 2 and history_signatures[-1] == history_signatures[-2])
        slot_summary = _build_slot_failure_summary(
            synthetic_failed,
            registry_entry,
            attempts=attempt_num,
            research_refetch_count=research_refetch_count,
            research_refetch_budget=research_refetch_budget,
            repeated=repeated,
            reason=last_failure_reason,
            failure_type=last_failure_type,
        )

        # Per-slot regen LLM call (nests under this slot's span automatically)
        try:
            decision = plan_one_slot_regen(
                slot_summary,
                generation_count=generation_count,
                retry_round=attempt_num,
                research_refetch_budget=research_refetch_budget,
                max_slot_regen_attempts=max_attempts,
                invoke_config={
                    "run_name": f"deepagent.regen_planner.slot_{slot}",
                    "tags": ["deepagent", "regen_planner", f"slot_{slot}"],
                    "metadata": {
                        "generation_count": generation_count,
                        "slot": slot,
                        "attempt_num": attempt_num,
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[slot_unit] slot %s regen LLM raised %s; will giveup", slot, exc)
            decision = {"decision": "giveup", "rationale": f"regen LLM error: {exc}"}

        if decision.get("decision") == "giveup":
            break

        # Apply decision to current_item for next attempt. Reuse the
        # existing helper which sets repair_strategy/mode/etc.
        _apply_regen_decision_to_retry_item(
            current_item, decision,
            slot=slot,
            research_refetch_count=research_refetch_count,
            research_refetch_budget=research_refetch_budget,
        )
        if decision.get("decision") == "research_and_regenerate":
            research_refetch_count += 1
        # Carry latest validation feedback into next attempt's context
        current_item["validation_feedback"] = last_failure_reason
        current_item["failure_type"] = last_failure_type

    # Fell out of loop — exhausted attempts or giveup
    failed_candidate = dict(candidate_before or current_item)
    failed_candidate.update({
        "validation_feedback": last_failure_reason,
        "failure_signature": _reason_signature(last_failure_reason),
        "failure_type": last_failure_type or "regenerability_failure",
        "failure_stage": "slot_unit_loop",
        "failure_reason": last_failure_reason,
        "_slot": slot,
    })
    return {
        "slot_outputs": [
            _wrap_slot_output(
                generation_count, slot,
                failed=failed_candidate,
                kind="validated_fail",
                contract_repair_strategy="",
                validation_passed=False,
                validation_reason=last_failure_reason or "max_attempts_exhausted",
                slot_failure={
                    "failure_type": last_failure_type or "regenerability_failure",
                    "failure_stage": "slot_unit_loop",
                    "failure_reason": last_failure_reason,
                },
            )
        ]
    }


def slot_aggregate_node(state: AgentState) -> Dict[str, Any]:
    """Project this generation's slot_outputs into approved/failed shape,
    apply ALL cross-slot validate decisions (slot_conflict, elite_backfill,
    retry counter, prune/clear), and route directly to ground/save/regen.

    Phase B (Option B1) consolidates per-slot validate INTO slot_unit and
    cross-slot decisions HERE. The legacy `validate_candidates_node` is
    bypassed entirely on the slot fan-out path.
    """
    generation_count = int(state.get("generation_count", 0))
    raw_outputs = list(state.get("slot_outputs", []) or [])
    # NOTE: must NOT use `or -1` here — Python treats 0 as falsy, so
    # `int(o.get("generation", -1) or -1)` would map generation=0 to -1
    # and the filter would drop ALL gen-0 entries. Use explicit None check.
    def _gen_of(o: Dict) -> int:
        v = o.get("generation")
        try:
            return int(v) if v is not None else -1
        except (TypeError, ValueError):
            return -1
    relevant = [o for o in raw_outputs if _gen_of(o) == generation_count]

    # Project slot_outputs into the legacy state shape:
    #   - validated_pass / ready / survivor → candidates
    #   - validated_fail / failed_synthesis / failed_repair → failed_problems
    #   - slot_failure metadata → slot_failure_registry
    candidates: List[Dict[str, Any]] = []
    failed_problems: List[Dict[str, Any]] = list(state.get("failed_problems", []) or [])
    slot_failure_registry = _slot_failure_registry(state)
    lineage_metrics = dict(state.get("lineage_metrics", {}) or {})
    validation_evidence_map = dict(state.get("validation_evidence", {}) or {})
    feedback = list(state.get("validation_feedback", []) or [])
    # approved_by_slot accumulates passing candidates by slot; survivors auto-go here too.
    approved_by_slot: Dict[int, Dict[str, Any]] = {
        int(item.get("slot", -1)): item.get("problem")
        for item in (state.get("approved_candidates", []) or [])
        if item.get("problem") is not None
    }

    for output in relevant:
        slot = int(output.get("slot", 0) or 0)
        cand = output.get("candidate")
        failed = output.get("failed")
        slot_failure = output.get("slot_failure")
        validation_passed = output.get("validation_passed")
        validation_reason = str(output.get("validation_reason", "") or "")
        kind = output.get("kind", "")

        if cand:
            cand = _ensure_unique_problem_id(
                cand,
                state,
                existing_candidates=candidates,
                failed_problems=failed_problems,
            )
            if cand.get("id"):
                lineage_metrics[cand.get("id")] = cand.get("lineage_metrics", {})
                validation_evidence_map[cand.get("id")] = dict(cand.get("validation_evidence", {}) or {})
            candidates.append(cand)
            # Phase B routing: classify the candidate based on per-slot
            # validate result captured by slot_unit.
            if validation_passed is True:
                # Slot conflict: this slot already has an approved candidate
                # (typically a survivor that was auto-approved earlier).
                if slot in approved_by_slot:
                    conflict_reason = f"Slot conflict: slot {slot} already has an approved candidate."
                    failed_candidate = dict(cand)
                    failed_candidate["validation_feedback"] = conflict_reason
                    failed_candidate["failure_signature"] = _reason_signature(conflict_reason)
                    failed_candidate["failure_type"] = "slot_conflict"
                    failed_candidate["failure_stage"] = "slot_aggregate"
                    failed_candidate["failure_reason"] = conflict_reason
                    failed_problems.append(failed_candidate)
                    feedback.append(conflict_reason)
                    slot_failure_registry = _record_slot_failure(
                        slot_failure_registry,
                        slot,
                        failure_type="slot_conflict",
                        failure_stage="slot_aggregate",
                        failure_reason=conflict_reason,
                    )
                else:
                    approved_by_slot[slot] = cand
            elif validation_passed is False:
                # Per-slot validate hard-failed inside slot_unit.
                failed_candidate = dict(cand)
                failed_candidate["validation_feedback"] = validation_reason
                failed_candidate["failure_signature"] = _reason_signature(validation_reason)
                failed_candidate["failure_type"] = _failure_type_from_reason(validation_reason)
                failed_candidate["failure_stage"] = "slot_unit_validate"
                failed_candidate["failure_reason"] = validation_reason
                failed_problems.append(failed_candidate)
                feedback.append(validation_reason)
                slot_failure_registry = _record_slot_failure(
                    slot_failure_registry,
                    slot,
                    failure_type=failed_candidate["failure_type"],
                    failure_stage="slot_unit_validate",
                    failure_reason=validation_reason,
                )
                logger.warning("❌ %s: %s", cand.get("id", "unknown"), validation_reason)
            else:
                # validation_passed is None → legacy survivor or pre-Phase-B
                # output. Treat as auto-approved (matches old behaviour).
                if slot in approved_by_slot:
                    # Same slot conflict semantics as above (rare for survivors).
                    conflict_reason = f"Slot conflict: slot {slot} already has an approved candidate."
                    failed_candidate = dict(cand)
                    failed_candidate["validation_feedback"] = conflict_reason
                    failed_candidate["failure_signature"] = _reason_signature(conflict_reason)
                    failed_candidate["failure_type"] = "slot_conflict"
                    failed_candidate["failure_stage"] = "slot_aggregate"
                    failed_candidate["failure_reason"] = conflict_reason
                    failed_problems.append(failed_candidate)
                    feedback.append(conflict_reason)
                else:
                    approved_by_slot[slot] = cand
        if failed:
            failed_problems.append(failed)
            if not validation_reason:
                validation_reason = str(failed.get("failure_reason", "") or failed.get("validation_feedback", "") or "")
            if validation_reason:
                feedback.append(validation_reason)
        if slot_failure:
            slot_failure_registry = _record_slot_failure(
                slot_failure_registry,
                slot,
                failure_type=slot_failure.get("failure_type", "regenerability_failure"),
                failure_stage=slot_failure.get("failure_stage", "slot_unit"),
                failure_reason=slot_failure.get("failure_reason", ""),
                repair_strategy=slot_failure.get("repair_strategy", ""),
            )

    # Cross-slot peer context attachment for any candidates that need it
    # downstream (ground_and_rescore expects orchestrator_peer_context).
    candidates = _attach_peer_context(
        candidates,
        phase="slot_aggregate",
        failed_problems=failed_problems,
    )

    # ── Cross-slot validate decisions (migrated from validate_candidates_node) ──
    params = dict(state.get("parameters", {}) or {})
    required_population = _desired_generation_size(state, params)
    min_survivable_population = max(1, min(int(params.get("min_survivable_population", 3) or 3), required_population))
    failed_problems = _prune_failed_for_approved_slots(approved_by_slot, failed_problems)
    failed_problems = _clear_nonblocking_failed_when_population_full(approved_by_slot, failed_problems, required_population)
    approved = [{"slot": slot, "problem": problem} for slot, problem in sorted(approved_by_slot.items(), key=lambda item: item[0])]
    total_valid = len(approved_by_slot)

    # Initial routing decision (Phase G: ground_and_rescore is done inside
    # slot_unit per-slot, so the cross-slot ground_and_rescore_node is
    # bypassed — go straight to save_generation when all slots passed).
    if not failed_problems and total_valid == required_population:
        next_phase = "save_generation"
    else:
        next_phase = "save_generation"  # post-elite_backfill default; partial save

    # Phase F: per-slot retries are exhausted INSIDE slot_unit before
    # slot_aggregate runs. There is no cross-slot retry round anymore — so
    # we skip the `_validation_retry_count` accumulator and fire
    # elite_backfill IMMEDIATELY when any slot is missing. This guarantees
    # no infinite cross-slot retry loop and the slot_unit retry budget is
    # the only retry mechanism.
    retry_count = int(params.get("_validation_retry_count", 0))
    params["_validation_retry_count"] = retry_count + 1 if (failed_problems or total_valid < required_population) else 0
    max_retries = int(params.get("max_validation_retries", 2))

    fallback_events: List[Dict[str, Any]] = []
    # Phase F: trigger elite_backfill on the FIRST aggregate run with
    # missing slots (no cross-slot retry round preceding it).
    if failed_problems or total_valid < required_population:
        approved, failed_problems, fallback_events = _select_elite_backfill(state, approved, failed_problems)
        approved_by_slot = {
            int(item.get("slot", -1)): item.get("problem")
            for item in approved
            if item.get("problem") is not None
        }
        failed_problems = _prune_failed_for_approved_slots(approved_by_slot, failed_problems)
        failed_problems = _clear_nonblocking_failed_when_population_full(approved_by_slot, failed_problems, required_population)
        total_valid = len(approved_by_slot)
        if fallback_events:
            logger.warning(
                "⚠️ Elite backfill activated for slots %s",
                ", ".join(str(event["slot"]) for event in fallback_events),
            )
            if not failed_problems and total_valid == required_population:
                next_phase = "ground_and_rescore"
        if failed_problems or total_valid < required_population:
            params["elite_backfill_attempted"] = True
            params["elite_backfill_count"] = len(fallback_events)
            if total_valid >= min_survivable_population:
                failed_problems = _prune_failed_for_approved_slots(approved_by_slot, failed_problems)
                next_phase = "save_generation"
            else:
                # Phase F: per-slot retries already exhausted inside slot_unit
                # AND elite_backfill couldn't reach min_survivable_population.
                # No cross-slot retry round to fall back to — hard error.
                raise RuntimeError(
                    f"Per-slot retries + elite_backfill exhausted; only "
                    f"{total_valid}/{required_population} slots valid (min survivable {min_survivable_population}). "
                    f"{len(failed_problems)} failures remain."
                )
    params["elite_backfill_count"] = len(fallback_events)
    failed_problems = _prune_failed_for_approved_slots(approved_by_slot, failed_problems)
    if total_valid >= required_population:
        failed_problems = _clear_nonblocking_failed_when_population_full(approved_by_slot, failed_problems, required_population)

    approved_problems = [problem for _slot, problem in sorted(approved_by_slot.items(), key=lambda item: item[0])]
    approved_problems = _attach_peer_context(
        approved_problems,
        phase="slot_aggregate",
        failed_problems=failed_problems,
    )
    approved = [{"slot": int(p.get("_slot", p.get("slot", 0)) or 0), "problem": p} for p in approved_problems]

    # Final routing override — Phase G: ground done in slot_unit, so success
    # path goes directly to save_generation. Partial save also goes to save.
    if not failed_problems and total_valid == required_population:
        next_phase = "save_generation"
    elif total_valid >= min_survivable_population:
        next_phase = "save_generation"

    # Build the matching ToolMessage for each slot's outcome (closes the
    # generator.dispatch tool_call envelope opened in slot_dispatch_node).
    tool_call_id_map: Dict[str, str] = (
        (state.get("run_metadata", {}) or {}).get("slot_dispatch_tool_call_ids", {}) or {}
    )
    slot_outcome: Dict[int, Dict[str, Any]] = {}
    for entry in approved:
        cand = entry.get("problem") or {}
        slot = int(entry.get("slot", 0) or 0)
        if cand.get("type") == "survivor":
            continue
        slot_outcome[slot] = {
            "slot": slot,
            "status": "candidate_ready",
            "candidate_id": cand.get("id", ""),
            "op_type": cand.get("op_type", ""),
            "pair_id": cand.get("pair_id"),
        }
    for failed in failed_problems:
        slot = int(failed.get("_slot", failed.get("slot", 0)) or 0)
        slot_outcome.setdefault(
            slot,
            {
                "slot": slot,
                "status": "failed",
                "failure_type": failed.get("failure_type", ""),
                "failure_stage": failed.get("failure_stage", ""),
                "op_type": failed.get("op_type", ""),
                "pair_id": failed.get("pair_id"),
            },
        )
    dispatch_tool_messages: List[Any] = []
    for slot, outcome in sorted(slot_outcome.items()):
        tool_call_id = tool_call_id_map.get(str(slot), "")
        dispatch_tool_messages.append(
            wrap_dispatch_result(tool_call_id, "generator.dispatch", outcome)
        )

    summary_msg = SystemMessage(
        content=(
            f"Slot fan-out aggregate: {total_valid} approved, {len(failed_problems)} failed → {next_phase}."
        )
    )

    result = {
        "session_phase": next_phase,
        "approved_candidates": approved,
        "candidates": [],  # consumed — downstream uses approved_candidates
        "failed_problems": failed_problems,
        "valid_count": total_valid,
        "generation_save_status": "partial_save" if next_phase == "save_generation" and total_valid < required_population else "complete",
        "validation_feedback": feedback,
        "validation_evidence": validation_evidence_map,
        "lineage_metrics": lineage_metrics,
        "slot_failure_registry": slot_failure_registry,
        "slot_retry_registry": slot_failure_registry,
        "retry_registry": slot_failure_registry,
        "parameters": params,
        "messages": dispatch_tool_messages + [summary_msg],
    }
    _write_json_artifact(
        {**state, **result},
        "slot_aggregate",
        {
            "generation_count": generation_count,
            "approved_count": total_valid,
            "failed_problems_count": len(failed_problems),
            "slot_outputs_count": len(relevant),
            "next_phase": next_phase,
        },
    )
    return result


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────


def _wrap_slot_output(
    generation_count: int,
    slot: int,
    *,
    candidate: Dict[str, Any] = None,
    failed: Dict[str, Any] = None,
    slot_failure: Dict[str, Any] = None,
    kind: str = "ready",
    contract_repair_strategy: str = "",
    validation_passed: Optional[bool] = None,
    validation_reason: str = "",
) -> Dict[str, Any]:
    """Build a single slot_outputs entry. Tagged with `generation` so the
    aggregator can filter to the current generation.

    `validation_passed` and `validation_reason` are populated by Phase B
    (per-slot validate inside slot_unit). When `validation_passed=False`, the
    aggregator routes the candidate into failed_problems with the reason as
    failure_feedback. When `validation_passed=True`, the candidate becomes a
    candidate for approved_by_slot. When `validation_passed=None` (e.g.
    survivor or pre-Phase-B output), the legacy `kind == "ready"` branch
    treats it as approved.
    """
    return {
        "generation": int(generation_count),
        "slot": int(slot),
        "kind": kind,
        "candidate": candidate,
        "failed": failed,
        "slot_failure": slot_failure,
        "contract_repair_strategy": contract_repair_strategy,
        "validation_passed": validation_passed,
        "validation_reason": validation_reason,
    }
