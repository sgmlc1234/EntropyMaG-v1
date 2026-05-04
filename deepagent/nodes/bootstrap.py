"""Bootstrap nodes: load_or_resume and init_run_memory."""

from typing import Dict, List

from langchain_core.messages import SystemMessage

from data_paths import DEFAULT_GENERATION_FORMAT, load_latest_generation, load_seed_problems
from deepagent.graph.artifacts import _write_json_artifact
from deepagent.graph.helpers import _lineage_index
from deepagent.graph.runtime import (
    _canonical_target_generation_size,
    _cumulative_validated_generated_count,
    _normalized_parameters,
)
from deepagent.invariants import extract_invariant_bundles
from deepagent.memory_bank import load_archival_memory_handle
from deepagent.state_full import AgentState


def load_or_resume_node(state: AgentState):
    options = state.get("session_options", {}) or {}
    save_format = options.get("save_format", DEFAULT_GENERATION_FORMAT)
    use_latest = bool(options.get("use_latest"))
    current = state.get("current_generation")
    gen_count = state.get("generation_count", 0)
    latest_file = None
    load_source = "existing_state"

    if not current:
        if use_latest:
            current, gen_count, latest_file = load_latest_generation(save_format=save_format)
            if current:
                load_source = "resume_latest"
        if not current:
            current = load_seed_problems(options.get("seed_spec"))
            gen_count = 0
            load_source = "seed_bootstrap"

    params = _normalized_parameters({**state, "generation_count": gen_count})
    messages = []
    if latest_file:
        messages.append(SystemMessage(content=f"Loaded latest generation from {latest_file}"))
    else:
        messages.append(SystemMessage(content=f"Loaded {len(current)} seed problems"))
    result = {
        "session_initialized": True,
        "session_phase": "init_run_memory",
        "current_generation": current,
        "current_generation_size": len(current),
        "generation_count": gen_count,
        "target_generation_size": _canonical_target_generation_size(params),
        "desired_generation_size": state.get("desired_generation_size", _canonical_target_generation_size(params)),
        "cumulative_validated_generated_count": _cumulative_validated_generated_count(state),
        "stop_reason": state.get("stop_reason", ""),
        "parameters": params,
        "approved_candidates": [],
        "failed_problems": [],
        "candidates": [],
        "work_items": [],
        "pair_plans": [],
        "pair_results": {},
        "pair_health_scores": {},
        "mutation_policies": {},
        "synthesis_briefs": {},
        "lineage_metrics": _lineage_index(current),
        "invariant_bundles": {},
        "run_working_memory": {},
        "archival_memory_handle": {},
        "validation_evidence": {},
        "context_packs": {},
        "research_artifacts": [],
        "retry_registry": {},
        "slot_retry_registry": {},
        "slot_failure_registry": {},
        "validation_feedback": [],
        "repair_queue": [],
        "generation_handoff_ready": False,
        "load_source": load_source,
        "messages": messages,
    }
    _write_json_artifact(
        result,
        "load_or_resume",
        {
            "generation_count": gen_count,
            "latest_file": latest_file,
            "loaded_problem_ids": [problem.get("id") for problem in current],
        },
    )
    return result


def init_run_memory_node(state: AgentState):
    current_generation = state.get("current_generation", []) or []
    invariant_bundles = extract_invariant_bundles(current_generation)
    archival_handle = load_archival_memory_handle(state.get("session_options", {}) or {})
    result = {
        "session_phase": "plan_generation",
        "invariant_bundles": invariant_bundles,
        "archival_memory_handle": archival_handle,
        "run_working_memory": {},
        "current_generation_size": len(current_generation),
        # Reset the plan-outcome bridging snapshots at the start of each
        # generation; the next synthesis_plan_node / save_generation_node pair
        # will repopulate them.
        "planned_work_items": [],
        "last_work_items": [],
        "last_failed_problems": [],
        "last_saved_slot_map": {},
        "messages": [SystemMessage(content=f"Initialized run memory with {len(archival_handle.get('problem_cards', []))} archival problem cards.")],
    }
    _write_json_artifact(
        {**state, **result},
        "init_run_memory",
        {
            "generation_count": state.get("generation_count", 0),
            "archive_metrics": archival_handle.get("metrics", {}),
            "archive_paths": archival_handle.get("paths", {}),
        },
    )
    return result
