"""Artifact slimming and JSON artifact I/O extracted from graph_full.py."""

import json
import os
from typing import Any, Dict

from data_paths import ensure_parent_dir
from deepagent.state_full import AgentState


def _slim_problem(problem: Dict) -> Dict:
    return {
        "id": problem.get("id", ""),
        "slot": problem.get("_slot", problem.get("slot", 0)),
        "type": problem.get("type", ""),
        "op_type": problem.get("op_type", ""),
        "pair_id": problem.get("pair_id"),
        "parent_ids": list(problem.get("parent_ids", []) or []),
        "context_pack_digest": problem.get("context_pack_digest", ""),
        "worker_status": problem.get("worker_status", ""),
        "worker_failure_type": problem.get("worker_failure_type", ""),
        "worker_failure_stage": problem.get("worker_failure_stage", ""),
        "code_runtime_mode": problem.get("code_runtime_mode", ""),
        "validation_evidence_digest": ((problem.get("validation_evidence") or {}).get("evidence_digest", "")),
    }


def _slim_failed_problem(problem: Dict) -> Dict:
    return {
        "id": problem.get("id", ""),
        "slot": problem.get("_slot", problem.get("slot", 0)),
        "op_type": problem.get("op_type", ""),
        "pair_id": problem.get("pair_id"),
        "failure_type": problem.get("failure_type", ""),
        "failure_stage": problem.get("failure_stage", ""),
        "failure_reason": problem.get("failure_reason", problem.get("validation_feedback", "")),
        "repair_strategy": problem.get("repair_strategy", ""),
    }


def _slim_research_artifact(artifact: Dict) -> Dict:
    return {
        "slot": artifact.get("slot"),
        "pair_id": artifact.get("pair_id"),
        "query": artifact.get("query", ""),
        "tool_used": artifact.get("tool_used", ""),
        "evidence_block_count": int(artifact.get("evidence_block_count", 0) or 0),
        "source_count": len(artifact.get("sources", []) or []),
        "degraded": artifact.get("degraded", False),
        "sources": [
            {
                "title": source.get("title"),
                "url": source.get("url"),
                "source_type": source.get("source_type"),
            }
            for source in (artifact.get("sources", []) or [])[:3]
        ],
    }


def _slim_brief(brief: Dict) -> Dict:
    return {
        "pair_id": brief.get("pair_id", ""),
        "slot": brief.get("slot", 0),
        "brief_role": brief.get("brief_role", ""),
        "research_value": brief.get("research_value", ""),
        "allowed_claims": list(brief.get("allowed_claims", []) or [])[:3],
        "research_usage_expectation": brief.get("research_usage_expectation", ""),
        "bridge_feasibility_note": brief.get("bridge_feasibility_note", ""),
        "target_quantity_guard": brief.get("target_quantity_guard", ""),
        "relation_guard": brief.get("relation_guard", ""),
        "concept_to_activate": brief.get("concept_to_activate", ""),
        "pattern_to_avoid": brief.get("pattern_to_avoid", ""),
        "preferred_composition_pattern": brief.get("preferred_composition_pattern", ""),
        "parameter_reuse_policy": brief.get("parameter_reuse_policy", ""),
        "deep_variant_requirement": brief.get("deep_variant_requirement", ""),
        "retry_focus": brief.get("retry_focus", ""),
    }


def _slim_work_item(item: Dict) -> Dict:
    return {
        "slot": item.get("slot", 0),
        "pair_id": item.get("pair_id"),
        "op_type": item.get("op_type", ""),
        "parent_ids": list(item.get("parent_ids", []) or []),
        "mode": item.get("mode", ""),
        "variation_axis": item.get("variation_axis", ""),
        "seed_focus": item.get("seed_focus", ""),
        "query_hint": item.get("query_hint", ""),
        "execution_group": item.get("execution_group", ""),
        "requires_research": bool(item.get("requires_research", False)),
        "requires_llm_brief": bool(item.get("requires_llm_brief", False)),
        "context_pack_digest": item.get("context_pack_digest", ""),
    }


def _artifact_dir(state: AgentState) -> str:
    options = state.get("session_options", {}) or {}
    return options.get("artifact_dir", "")


def _artifact_path(state: AgentState, phase: str) -> str:
    artifact_dir = _artifact_dir(state)
    if not artifact_dir:
        return ""
    gen = state.get("generation_count", 0)
    return os.path.join(artifact_dir, f"{gen:03d}_{phase}.json")


def _artifact_detail_mode(state: AgentState) -> str:
    return str((state.get("session_options", {}) or {}).get("artifact_detail", "slim")).lower()


def _slim_artifact_payload(phase: str, payload: Dict) -> Dict:
    if phase == "init_run_memory":
        return payload
    if phase == "plan_generation":
        strategy = dict(payload.get("selection_strategy", {}) or {})
        return {
            "generation_count": payload.get("generation_count", 0),
            "selection_strategy": {
                "normalization_mode": strategy.get("normalization_mode", ""),
                "desired_generation_size": strategy.get("desired_generation_size", 0),
                "target_generation_size": strategy.get("target_generation_size", 0),
                "active_pool_ids": strategy.get("active_pool_ids", []),
                "plan_rationale": strategy.get("plan_rationale", ""),
                "strategy_source": strategy.get("strategy_source", ""),
                "op_type_allocation": strategy.get("op_type_allocation", {}),
            },
            "work_items": [_slim_work_item(item) for item in (payload.get("work_items", []) or [])],
            "pair_plans": [
                {
                    "pair_id": plan.get("pair_id"),
                    "parent_ids": plan.get("parent_ids", []),
                    "mutation_parent_id": plan.get("mutation_parent_id"),
                    "crossover_slot": plan.get("crossover_slot", 0),
                    "mutation_slot": plan.get("mutation_slot", 0),
                    "variation_axis": plan.get("variation_axis", ""),
                }
                for plan in (payload.get("pair_plans", []) or [])
            ],
        }
    if phase == "synthesis_plan":
        return {
            "generation_count": payload.get("generation_count", 0),
            "synthesis_plans": payload.get("synthesis_plans", {}),
        }
    if phase == "research_candidates":
        return {
            "generation_count": payload.get("generation_count", 0),
            "research_artifacts": [_slim_research_artifact(item) for item in (payload.get("research_artifacts", []) or [])],
        }
    if phase == "prepare_synthesis_briefs":
        return {
            "generation_count": payload.get("generation_count", 0),
            "synthesis_briefs": {key: _slim_brief(value) for key, value in (payload.get("synthesis_briefs", {}) or {}).items()},
        }
    if phase in {"synthesize_candidates", "repair_failed_candidates", "postprocess_candidates", "validate_candidates"}:
        return {
            "generation_count": payload.get("generation_count", 0),
            "valid_count": payload.get("valid_count", 0),
            "non_survivor_valid_count": payload.get("non_survivor_valid_count", 0),
            "candidates": [_slim_problem(item) for item in (payload.get("candidates", []) or [])],
            "approved_candidates": [
                {"slot": item.get("slot", 0), "problem": _slim_problem(item.get("problem", {}) or {})}
                for item in (payload.get("approved_candidates", []) or [])
            ],
            "failed_problems": [_slim_failed_problem(item) for item in (payload.get("failed_problems", []) or [])],
            "repair_queue": [
                {
                    "slot": item.get("slot", 0),
                    "pair_id": item.get("pair_id"),
                    "op_type": item.get("op_type", ""),
                    "repair_strategy": item.get("repair_strategy", ""),
                    "failure_type": item.get("failure_type", ""),
                }
                for item in (payload.get("repair_queue", []) or [])
            ],
            "validation_feedback": payload.get("validation_feedback", []),
            "elite_backfill_events": payload.get("elite_backfill_events", []),
            "execution_trace": payload.get("execution_trace", {}),
        }
    if phase == "save_generation":
        return {
            "generation_count": payload.get("generation_count", 0),
            "output_file": payload.get("output_file", ""),
            "current_generation": [problem.get("id") for problem in (payload.get("current_generation", []) or [])],
            "pair_results": payload.get("pair_results", {}),
            "pair_health_scores": payload.get("pair_health_scores", {}),
            "mutation_policies": payload.get("mutation_policies", {}),
        }
    if phase == "consolidate_archive":
        return {
            "generation_count": payload.get("generation_count", 0),
            "archive_metrics": payload.get("archive_metrics", {}),
            "archive_paths": payload.get("archive_paths", {}),
        }
    if phase == "review_generation":
        return {
            "generation_count": payload.get("generation_count", 0),
            "review_prompt": payload.get("review_prompt", ""),
            "current_generation": [problem.get("id") for problem in (payload.get("current_generation", []) or [])],
        }
    if phase == "regenerate_failed":
        # Phase D3d (2026-04-16): the retry artifact previously dumped each
        # retry_item fully (parents, context_pack, candidate_before_repair)
        # averaging 60KB. Keep only the decision trail.
        return {
            "generation_count": payload.get("generation_count", 0),
            "retry_round": payload.get("retry_round", 0),
            "research_refetch_budget": payload.get("research_refetch_budget", 0),
            "regen_plan": payload.get("regen_plan", {}),
            "retry_items": [
                {
                    "slot": item.get("slot", 0),
                    "op_type": item.get("op_type", ""),
                    "pair_id": item.get("pair_id"),
                    "failure_signature": item.get("failure_signature", ""),
                    "failure_type": item.get("failure_type", ""),
                    "requires_research": bool(item.get("requires_research", False)),
                    "repair_strategy": item.get("repair_strategy", ""),
                    "retry_feedback_summary": item.get("retry_feedback_summary", ""),
                    "_regen_decision": item.get("_regen_decision", ""),
                    "research_refetch_count": item.get("research_refetch_count", 0),
                }
                for item in (payload.get("retry_items", []) or [])
            ],
        }
    return payload


def _write_json_artifact(state: AgentState, phase: str, payload: Dict):
    path = _artifact_path(state, phase)
    if not path:
        return
    ensure_parent_dir(path)
    if _artifact_detail_mode(state) != "full":
        payload = _slim_artifact_payload(phase, payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
