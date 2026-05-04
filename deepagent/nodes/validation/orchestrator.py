"""Validation-phase nodes: postprocess, validate, ground_and_rescore."""

import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.messages import SystemMessage

from deepagent.grounding import ground_and_rescore
from deepagent.invariants import audit_candidate_against_bundles
from deepagent.memory_bank import retrieve_archival_evidence, retrieve_topk_archive_by_text
from deepagent.quality import assess_problem_quality
from deepagent.tracing import record_deterministic_span
from deepagent.state_full import AgentState
from deepagent.nodes.validation.worker import (
    assess_near_copy,
    assess_solvability,
    build_solvability_ablation_skip,
    build_execution_evidence,
    ground_solution,
    materialize_answer_from_code,
    validate_problem,
    validate_problem_by_code,
)

from deepagent.graph.runtime import (
    MAX_PARALLEL_DISPATCH_FALLBACK,
    _canonical_target_generation_size,
    _coerce_difficulty_value,
    _desired_generation_size,
    _normalized_parameters,
)
from deepagent.graph.artifacts import _slim_problem, _slim_failed_problem, _write_json_artifact
from deepagent.graph.helpers import (
    _clear_nonblocking_failed_when_population_full,
    _failure_type_from_reason,
    _make_failed_problem,
    _prune_failed_for_approved_slots,
    _reason_signature,
    _record_slot_failure,
    _select_elite_backfill,
    _slot_failure_registry,
)
from deepagent.tracing import _dispatch_fifo_parallel

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


def _attach_peer_context_to_problems(
    problems: List[Dict[str, Any]],
    *,
    phase: str,
    failed_problems: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
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


def _assess_constraint_guard(candidate: Dict, parents: List[Dict], invariant_bundles: List[Dict]) -> Dict:
    """Deterministic invariant audit — inlined from constraint_guard.py."""
    audit = audit_candidate_against_bundles(candidate, invariant_bundles)
    if not all(
        [
            audit.get("definition_preserved", False),
            audit.get("relation_preserved", False),
        ]
    ):
        issue_types = []
        if not audit.get("definition_preserved", True):
            issue_types.append("definition_rewrite")
        if not audit.get("relation_preserved", True):
            issue_types.append("relation_rewrite")
        reason = "; ".join(audit.get("reasons", [])) or "Invariant audit failed."
        return {
            "passed": False,
            "issue_types": issue_types or ["other"],
            "reason": reason,
            "deterministic": True,
            "audit": audit,
        }
    return {
        "passed": True,
        "issue_types": [],
        "reason": "Deterministic invariant audit passed.",
        "deterministic": True,
        "audit": audit,
    }


def validate_one_candidate(
    candidate: Dict,
    *,
    invariant_bundles_for_parents: Dict[str, Dict],
    parents_for_candidate: Dict[str, Dict],
    archival_memory_handle: Dict,
    generation_count: int,
    ablation_condition: str = "full",
) -> Tuple[bool, str]:
    """Per-slot validation pipeline — extracted from `_validation_worker` closure
    inside `validate_candidates_node`. Pure function (no closure deps): all
    state subset is passed explicitly so the function can be invoked from
    `slot_unit_node` (Send branch) where full AgentState is unavailable.

    Mutates the candidate dict in place to attach assessment evidence
    (`near_copy_assessment`, `constraint_guard_assessment`, `invariant_audit`,
    `solvability_*`, `validation_evidence`, `validation_assessment`,
    `quality_assessment`, ...) — same side effects as the original closure.

    Returns `(passed, reason)` tuple matching the closure's contract.
    """
    ablation_condition = str(ablation_condition or "full").strip() or "full"
    if ablation_condition not in {"full", "no_near_copy", "no_solvability"}:
        ablation_condition = "full"
    candidate["ablation_condition"] = ablation_condition

    stmt = (candidate.get("statement") or "").strip()
    ans = (candidate.get("answer") or "").strip()
    sol = (candidate.get("solution") or candidate.get("solution_sketch") or "").strip()
    slot = candidate.get("_slot", -1)
    invoke_metadata = {
        "slot": slot,
        "pair_id": candidate.get("pair_id"),
        "problem_id": candidate.get("id", ""),
        "sandbox_mode": candidate.get("code_runtime_mode", ""),
        "sandbox_python_bin": candidate.get("sandbox_python_bin", ""),
        "code_gate_error_type": candidate.get("code_gate_error_type", ""),
    }
    axis_used = str(candidate.get("variation_axis_used", "") or "").strip().lower()
    if axis_used in {"axis_missing", "bridge_missing"}:
        signal_name = "axis_missing" if axis_used == "axis_missing" else "bridge_missing"
        candidate[f"{signal_name}_signal"] = True
        record_deterministic_span(
            f"validator.{signal_name}_abort",
            inputs={
                "problem_id": candidate.get("id", ""),
                "slot": slot,
                "op_type": candidate.get("op_type", ""),
            },
            outputs={
                "verdict": "reject",
                "reason": (
                    "axis_missing signal emitted by generator — planner produced empty variation_axis"
                    if axis_used == "axis_missing"
                    else "bridge_missing signal emitted by crossover generator — no genuine two-parent bridge was available"
                ),
            },
            tags=["deepagent", "validator", signal_name],
        )
        return False, (
            "axis_missing signal: dispatch provided no variation_axis. "
            "Orchestrator should replan this slot before retry."
            if axis_used == "axis_missing"
            else "bridge_missing signal: crossover generator could not state a genuine shared invariant. "
            "Orchestrator should reject or replan this slot before retry."
        )
    if not (stmt and ans and sol):
        return False, "Missing statement/answer/solution"
    code_passed, code_reason = validate_problem_by_code(candidate)
    if not code_passed:
        return False, f"Correctness gate failed: {code_reason}"
    parent_ids = candidate.get("parent_ids", []) or []
    invariant_bundles = [
        invariant_bundles_for_parents.get(pid)
        for pid in parent_ids
        if invariant_bundles_for_parents.get(pid)
    ]
    invariant_parents = [
        parents_for_candidate[pid]
        for pid in parent_ids
        if pid in parents_for_candidate
    ]
    near_copy_assessment = assess_near_copy(candidate, invariant_parents)
    candidate["near_copy_assessment"] = near_copy_assessment
    near_copy_gate_enforced = ablation_condition != "no_near_copy"
    candidate["near_copy_gate_enforced"] = near_copy_gate_enforced
    record_deterministic_span(
        "validator.near_copy_check",
        inputs={
            "child_id": candidate.get("id", ""),
            "parent_ids": [p.get("id", "") for p in invariant_parents],
            "child_answer": str(candidate.get("answer", ""))[:60],
            "similarity_threshold": 0.85,
            "method": "sequence_matcher_on_normalized_statement",
        },
        outputs={
            "near_copy": near_copy_assessment.get("near_copy"),
            "similarity": near_copy_assessment.get("similarity"),
            "matched_parent_id": near_copy_assessment.get("matched_parent_id", ""),
            "reason": near_copy_assessment.get("reason", ""),
            "gate_enforced": near_copy_gate_enforced,
        },
        tags=["deepagent", "validator", "near_copy"],
        metadata={"slot": slot, "problem_id": candidate.get("id", "")},
    )
    if near_copy_assessment.get("near_copy") and near_copy_gate_enforced:
        return False, f"Near-copy of parent blocked: {near_copy_assessment.get('reason', '')}"
    if near_copy_assessment.get("near_copy") and not near_copy_gate_enforced:
        candidate["near_copy_ablation_bypassed"] = True
    invariant_assessment = _assess_constraint_guard(candidate, invariant_parents, invariant_bundles)
    candidate["constraint_guard_assessment"] = invariant_assessment
    candidate["invariant_audit"] = invariant_assessment.get("audit", candidate.get("invariant_audit"))
    if not invariant_assessment.get("passed"):
        guard_reason = invariant_assessment.get("reason", "Constraint guard advisory.")
        return False, guard_reason
    solvability_gate_enforced = ablation_condition != "no_solvability"
    candidate["solvability_gate_enforced"] = solvability_gate_enforced
    if solvability_gate_enforced:
        solvability_assessment = assess_solvability(
            candidate,
            invariant_bundles=invariant_bundles,
            invoke_config={
                "run_name": f"deepagent.solvability.slot_{slot}",
                "tags": ["deepagent", "solvability"],
                "metadata": invoke_metadata,
            },
        )
    else:
        solvability_assessment = build_solvability_ablation_skip(
            candidate,
            invariant_bundles=invariant_bundles,
        )
    candidate["solvability_gate_required"] = bool(solvability_assessment.get("gate_required", False))
    candidate["solvability_gate_family"] = solvability_assessment.get("constraint_family", "")
    candidate["solvability_assessment"] = solvability_assessment
    candidate["solvability_failure_type"] = solvability_assessment.get("failure_type", "")
    candidate["solvability_feedback"] = solvability_assessment.get("reason", "")
    if solvability_assessment.get("verdict") == "fail" and solvability_gate_enforced:
        return False, f"Solvability failed: {solvability_assessment.get('reason', 'Constraint system is inconsistent.')}"
    archival_evidence = retrieve_archival_evidence(
        candidate,
        archival_memory_handle or {},
        generation_count=generation_count,
    )
    candidate["validation_evidence"] = archival_evidence
    verdict, reason, validation_assessment = validate_problem(
        candidate,
        invariant_bundles=invariant_bundles,
        archival_evidence=archival_evidence,
        parents=invariant_parents,
        invoke_config={
            "run_name": f"deepagent.validator.slot_{slot}",
            "tags": ["deepagent", "validator"],
            "metadata": invoke_metadata,
        },
    )
    candidate["validation_assessment"] = validation_assessment
    if verdict == "advisory_fail":
        candidate["validation_advisory"] = reason
    if verdict == "hard_fail":
        return False, reason
    parents = list(parents_for_candidate.values())
    target_diff = _coerce_difficulty_value(candidate.get("difficulty", 0), default=0.0)
    quality_passed, quality_reason, quality_assessment = assess_problem_quality(
        candidate,
        parents,
        target_diff,
        invoke_config={
            "run_name": f"deepagent.quality.slot_{slot}",
            "tags": ["deepagent", "quality"],
            "metadata": invoke_metadata,
        },
    )
    candidate["quality_assessment"] = quality_assessment
    if not quality_passed:
        candidate["quality_advisory"] = quality_reason
    return True, reason


def run_one_ground_and_rescore(
    candidate: Dict,
    *,
    parents_for_candidate: Dict[str, Dict],
    synthesis_brief: Dict,
    sandbox_evidence: Any,
    archival_memory_handle: Dict,
    op_type: str,
) -> Dict[str, Any]:
    """Per-candidate grounding + adversarial probe + difficulty rescore.

    Phase G: extracted from `_grounding_worker` closure inside
    `ground_and_rescore_node` so `slot_unit_node` can call this inline
    after a validate pass — making `ground_and_rescore.slot_N` LLM spans
    children of `generator.operation.slot_{slot}` and letting slots that
    pass both validate AND ground exit immediately (no waiting for the
    cross-slot ground_and_rescore_node aggregator).

    Mutates `candidate` in place (sets `ground_and_rescore_assessment`,
    overwrites `solution` on accept, sets `difficulty`/`difficulty_rescored`
    on accept).

    Returns the raw ground_and_rescore() result dict.
    """
    slot = candidate.get("_slot", candidate.get("slot", 0))
    try:
        slot_int = int(slot)
    except Exception:
        slot_int = -1
    parents = list(parents_for_candidate.values())
    sandbox_evidence_str = ""
    if isinstance(sandbox_evidence, dict) and sandbox_evidence:
        try:
            sandbox_evidence_str = json.dumps(sandbox_evidence, ensure_ascii=False)[:1600]
        except Exception:
            sandbox_evidence_str = str(sandbox_evidence)[:1600]
    elif isinstance(sandbox_evidence, str):
        sandbox_evidence_str = sandbox_evidence[:1600]
    archive_size = len(archival_memory_handle.get("problem_cards", []) or [])
    archive_neighbors = retrieve_topk_archive_by_text(candidate, archival_memory_handle or {}, k=3)
    record_deterministic_span(
        "grounding.archive_retrieval",
        run_type="retriever",
        inputs={
            "candidate_id": candidate.get("id", ""),
            "candidate_statement_preview": (candidate.get("statement") or "")[:240],
            "archive_size": archive_size,
            "k": 3,
            "method": "jaccard_token_overlap",
        },
        outputs={
            "matched_count": len(archive_neighbors),
            "matches": [
                {
                    "card_id": c.get("problem_id", ""),
                    "similarity": c.get("_retrieval_similarity"),
                    "generation": c.get("generation"),
                    "statement_preview": (c.get("statement_excerpt", "") or "")[:120],
                }
                for c in archive_neighbors
            ],
        },
        tags=["deepagent", "grounding", "retrieval"],
        metadata={"slot": slot_int, "problem_id": candidate.get("id", "")},
    )
    result = ground_and_rescore(
        candidate,
        parents,
        synthesis_brief=synthesis_brief or {},
        sandbox_evidence=sandbox_evidence_str,
        op_type=op_type,
        archive_neighbors=archive_neighbors,
        invoke_config={
            "run_name": f"deepagent.ground_and_rescore.slot_{slot_int}",
            "tags": ["deepagent", "grounding"],
            "metadata": {
                "slot": slot_int,
                "problem_id": candidate.get("id", ""),
                "op_type": op_type,
                "archive_neighbor_count": len(archive_neighbors),
            },
        },
    )
    candidate["ground_and_rescore_assessment"] = {k: v for k, v in result.items() if k != "raw"}
    decision = result.get("decision", "accept")
    if decision == "accept":
        grounded = result.get("grounded_solution") or candidate.get("solution")
        if grounded:
            candidate["solution"] = grounded
        try:
            rescored_raw = result.get("rescored_difficulty", candidate.get("difficulty"))
            rescored = float(rescored_raw) if isinstance(rescored_raw, (int, float)) else _coerce_difficulty_value(rescored_raw, default=_coerce_difficulty_value(candidate.get("difficulty"), default=7.0))
            candidate["difficulty"] = rescored
            candidate["difficulty_rescored"] = True
        except Exception:
            pass
    return result


def ground_and_rescore_node(state: AgentState):
    """P4 — single-LLM grounding + adversarial probe + difficulty rescore.

    Runs after validate_candidates produced approved candidates. For each
    approved candidate, calls the GroundingReviewSchema-backed reviewer once.
    Outcomes:
      - accept   : candidate stays approved; solution overwritten with the
                   grounded rewrite, difficulty replaced with rescored_difficulty.
      - rescope  : candidate moved to failed_problems with a rescope reason;
                   regenerate_failed will rebuild it.
      - reject   : same routing as rescope but with the reject reason.

    If the gate drains approved below min_survivable_population, route to
    regenerate_failed; otherwise proceed to save_generation.
    """
    approved = list(state.get("approved_candidates", []) or [])
    if not approved:
        # Nothing to review — preserve the upstream session_phase decision.
        return {"grounding_assessments": dict(state.get("grounding_assessments", {}) or {})}
    approved_problems = _attach_peer_context_to_problems(
        [dict(entry.get("problem") or {}) for entry in approved],
        phase="ground_and_rescore",
        failed_problems=list(state.get("failed_problems", []) or []),
    )
    approved = [
        {"slot": _problem_slot(problem), "problem": problem}
        for problem in approved_problems
    ]

    parents_by_id = {p.get("id"): p for p in (state.get("current_generation", []) or []) if p.get("id")}
    briefs = state.get("synthesis_briefs", {}) or {}
    state_validation_evidence = state.get("validation_evidence", {}) or {}
    # Build a slot→work_item map so we can recover op_type when the candidate
    # itself doesn't carry it (W4 fix).
    work_items_by_slot = {
        int(item.get("slot", -1)): item
        for item in (state.get("planned_work_items") or state.get("work_items") or [])
        if item.get("slot") is not None
    }

    grounding_assessments: Dict[str, Any] = dict(state.get("grounding_assessments", {}) or {})
    surviving: List[Dict] = []
    new_failed: List[Dict] = list(state.get("failed_problems", []) or [])
    rescope_reasons: List[str] = []
    max_parallel = int(_normalized_parameters(state).get("max_parallel_dispatch", MAX_PARALLEL_DISPATCH_FALLBACK))
    archive_handle = state.get("archival_memory_handle", {}) or {}

    # Survivors pass through; non-survivors are processed in parallel.
    survivor_entries = []
    pending_entries = []
    for entry in approved:
        problem = entry.get("problem") or {}
        problem_type = str(problem.get("type") or problem.get("op_type") or "").lower()
        if problem_type in {"survivor", "fallback_survivor"}:
            survivor_entries.append(entry)
        else:
            slot = entry.get("slot", problem.get("_slot", 0))
            entry_copy = dict(entry)
            entry_copy["_slot_trace_enabled"] = True
            entry_copy["_trace_name"] = f"orchestrator.ground_and_rescore.slot_{slot}"
            pending_entries.append(entry_copy)

    surviving.extend(survivor_entries)

    def _grounding_worker(entry: Dict) -> Dict:
        problem = dict(entry.get("problem") or {})
        slot = entry.get("slot", problem.get("_slot", 0))
        try:
            slot_int = int(slot)
        except Exception:
            slot_int = -1
        parent_ids = problem.get("parent_ids") or []
        parents = [parents_by_id[pid] for pid in parent_ids if pid in parents_by_id]
        synthesis_brief = briefs.get(slot) or briefs.get(slot_int) or briefs.get(str(slot)) or {}
        # Pull per-candidate sandbox evidence from the state-level map (C5 fix).
        sandbox_payload = (
            state_validation_evidence.get(problem.get("id"))
            or state_validation_evidence.get(slot_int)
            or state_validation_evidence.get(str(slot_int))
            or {}
        )
        sandbox_evidence_str = ""
        if isinstance(sandbox_payload, dict) and sandbox_payload:
            try:
                sandbox_evidence_str = json.dumps(sandbox_payload, ensure_ascii=False)[:1600]
            except Exception:
                sandbox_evidence_str = str(sandbox_payload)[:1600]
        elif isinstance(sandbox_payload, str):
            sandbox_evidence_str = sandbox_payload[:1600]
        # W4: recover op_type from work_item plan if the candidate is missing it.
        op_type = (
            str(problem.get("op_type") or problem.get("type") or "").strip()
            or str(work_items_by_slot.get(slot_int, {}).get("op_type", "")).strip()
        )
        archive_size = len(archive_handle.get("problem_cards", []) or [])
        archive_neighbors = retrieve_topk_archive_by_text(problem, archive_handle, k=3)
        record_deterministic_span(
            "grounding.archive_retrieval",
            run_type="retriever",
            inputs={
                "candidate_id": problem.get("id", ""),
                "candidate_statement_preview": (problem.get("statement") or "")[:240],
                "archive_size": archive_size,
                "k": 3,
                "method": "jaccard_token_overlap",
            },
            outputs={
                "matched_count": len(archive_neighbors),
                "matches": [
                    {
                        "card_id": c.get("problem_id", ""),
                        "similarity": c.get("_retrieval_similarity"),
                        "generation": c.get("generation"),
                        "statement_preview": (c.get("statement_excerpt", "") or "")[:120],
                    }
                    for c in archive_neighbors
                ],
            },
            tags=["deepagent", "grounding", "retrieval"],
            metadata={"slot": int(slot or 0), "problem_id": problem.get("id", "")},
        )
        result = ground_and_rescore(
            problem,
            parents,
            synthesis_brief=synthesis_brief,
            sandbox_evidence=sandbox_evidence_str,
            op_type=op_type,
            archive_neighbors=archive_neighbors,
            invoke_config={
                "run_name": f"deepagent.ground_and_rescore.slot_{slot}",
                "tags": ["deepagent", "grounding"],
                "metadata": {
                    "slot": int(slot or 0),
                    "problem_id": problem.get("id", ""),
                    "op_type": op_type,
                    "archive_neighbor_count": len(archive_neighbors),
                },
            },
        )
        problem["ground_and_rescore_assessment"] = {k: v for k, v in result.items() if k != "raw"}
        return {
            "entry": {**entry, "problem": problem},
            "result": result,
            "problem_key": problem.get("id", f"slot_{slot}"),
        }

    for _entry, worker_result in _dispatch_fifo_parallel(pending_entries, max_parallel, _grounding_worker, slot_trace_state=state):
        entry = worker_result["entry"]
        result = worker_result["result"]
        problem = entry.get("problem") or {}
        slot = entry.get("slot", problem.get("_slot", 0))
        grounding_assessments[worker_result["problem_key"]] = result
        decision = result.get("decision", "accept")
        if decision == "accept":
            grounded = result.get("grounded_solution") or problem.get("solution")
            if grounded:
                problem["solution"] = grounded
            try:
                rescored_raw = result.get("rescored_difficulty", problem.get("difficulty"))
                rescored = float(rescored_raw) if isinstance(rescored_raw, (int, float)) else _coerce_difficulty_value(rescored_raw, default=_coerce_difficulty_value(problem.get("difficulty"), default=7.0))
                problem["difficulty"] = rescored
                problem["difficulty_rescored"] = True
            except Exception:
                pass
            surviving.append(entry)
        else:
            reason = result.get("reason") or f"Grounding gate {decision}"
            failed_problem = dict(problem)
            failed_problem["failure_stage"] = "ground_and_rescore"
            failed_problem["failure_reason"] = f"[{decision}] {reason}"
            failed_problem["ground_and_rescore_assessment"] = problem.get("ground_and_rescore_assessment", {})
            if decision == "rescope" and result.get("rescope_suggestion"):
                failed_problem["rescope_suggestion"] = result.get("rescope_suggestion", "")
            new_failed.append(failed_problem)
            rescope_reasons.append(f"slot_{slot} {decision}: {reason}")

    params = _normalized_parameters(state)
    required_population = _desired_generation_size(state)
    min_survivable_population = max(1, min(int(params.get("min_survivable_population", 3) or 3), required_population))
    surviving_problems = _attach_peer_context_to_problems(
        [dict(entry.get("problem") or {}) for entry in surviving],
        phase="ground_and_rescore.accepted",
        failed_problems=new_failed,
    )
    surviving = [{"slot": _problem_slot(problem), "problem": problem} for problem in surviving_problems]

    if len(surviving) >= min_survivable_population:
        next_phase = "save_generation"
    else:
        next_phase = "regenerate_failed"

    feedback = list(state.get("validation_feedback", []) or []) + rescope_reasons

    result = {
        "session_phase": next_phase,
        "approved_candidates": surviving,
        "failed_problems": new_failed,
        "grounding_assessments": grounding_assessments,
        "validation_feedback": feedback,
        "messages": [
            SystemMessage(
                content=(
                    f"Grounding gate complete. Accept: {len(surviving)}, "
                    f"Rescope/Reject: {len(approved) - len(surviving)}"
                )
            )
        ],
    }
    _write_json_artifact(
        {**state, **result},
        "ground_and_rescore",
        {
            "generation_count": state.get("generation_count", 0),
            "accept_count": len(surviving),
            "rejected_or_rescoped": [
                {
                    "id": fp.get("id"),
                    "decision": (fp.get("ground_and_rescore_assessment") or {}).get("decision"),
                    "reason": fp.get("failure_reason"),
                }
                for fp in new_failed
                if fp.get("failure_stage") == "ground_and_rescore"
            ],
        },
    )
    return result
