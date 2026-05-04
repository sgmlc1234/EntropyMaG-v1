"""Graph routing functions.

Phase A/B/C cleanup: `_route_postprocess_phase` and `_route_validate_phase`
removed — their targets (`postprocess_candidates`, `validate_candidates`)
no longer exist. Cross-slot validate decisions moved into
`slot_aggregate_node`; that node's outgoing edge uses an inline lambda.
"""

from langgraph.graph import END
from deepagent.state_full import AgentState


def _continue_or_end(state: AgentState):
    phase = state.get("session_phase")
    if phase in {"awaiting_review", "generation_review_wait", "generation_handoff_ready"}:
        return END
    if phase in {"advisor_stage", "modifier_stage", "exit"}:
        return phase
    if phase in {"init_run_memory", "consolidate_archive", "save_generation", "review_generation"}:
        return END
    return "init_run_memory"


def _route_entry_phase(state: AgentState):
    """START routing — also handles resume of legacy session_phase values
    saved by older runs (synthesize_candidates / postprocess_candidates /
    validate_candidates / repair_failed_candidates) by mapping them to the
    slot fan-out path. The graph __init__ START conditional alias map
    points each legacy name to its slot-pipeline equivalent.
    """
    if not state.get("session_initialized"):
        return "load_or_resume"

    phase = state.get("session_phase") or "init_run_memory"
    if phase == "load_or_resume":
        return "init_run_memory"
    if phase in {"awaiting_review", "generation_review_wait", "review_generation"}:
        return "review_generation"
    if phase == "generation_handoff_ready":
        return "init_run_memory"
    if phase in {
        "init_run_memory",
        "plan_generation",
        "synthesis_plan",
        "research_candidates",
        "prepare_synthesis_briefs",
        # Slot fan-out names (current).
        "slot_dispatch",
        "slot_unit",
        "slot_aggregate",
        # Legacy aliases — mapped to slot pipeline by START conditional.
        "synthesize_candidates",
        "repair_failed_candidates",
        "postprocess_candidates",
        "validate_candidates",
        "ground_and_rescore",
        "regenerate_failed",
        "save_generation",
        "consolidate_archive",
        "advisor_stage",
        "modifier_stage",
        "exit",
    }:
        return phase
    return "init_run_memory"


def _route_context_phase(state: AgentState):
    phase = state.get("session_phase")
    if phase in {"synthesis_plan", "research_candidates", "prepare_synthesis_briefs", "synthesize_candidates"}:
        return phase
    return "research_candidates"


def _route_grounding_phase(state: AgentState):
    """Grounding gate decides save vs. regenerate based on whether the gate
    drained the approved slots below the survivable minimum."""
    return "save_generation" if state.get("session_phase") == "save_generation" else "regenerate_failed"
