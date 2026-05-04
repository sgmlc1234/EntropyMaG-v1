"""Graph wiring: create_deep_evolution_graph and nothing else.

Graph support modules live under ``deepagent.graph`` while tracing stays in
``deepagent.tracing``.

After Phase A/B/C cleanup the per-generation pipeline is purely slot fan-out:
  prepare_synthesis_briefs → slot_dispatch → [Send×N] → slot_unit
                                         → slot_aggregate → ground / save / regen

Legacy single-phase nodes (`synthesize_candidates`, `postprocess_candidates`,
`repair_failed_candidates`, `validate_candidates`) have been removed — their
responsibilities are split across `slot_unit` (per-slot synth+repair+postprocess+
validate) and `slot_aggregate` (cross-slot decisions).
"""
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from deepagent.state_full import AgentState

# ── Routing functions ───────────────────────────────────────────────────
from deepagent.graph.routing import (
    _continue_or_end,
    _route_context_phase,
    _route_entry_phase,
    _route_grounding_phase,
)

# ── Tracing infrastructure ──────────────────────────────────────────────
from deepagent.tracing import _trace_wrapped_node

# ── Node functions ──────────────────────────────────────────────────────
from deepagent.nodes.bootstrap import init_run_memory_node, load_or_resume_node
from deepagent.nodes.planning import (
    plan_generation_node,
    prepare_synthesis_briefs_node,
    research_candidates_node,
    synthesis_plan_node,
)
from deepagent.nodes.synthesis.slot_pipeline import (
    slot_aggregate_node,
    slot_dispatch_node,
    slot_dispatch_route,
    slot_unit_node,
)
from deepagent.nodes.validation import (
    ground_and_rescore_node,
)
from deepagent.nodes.persistence import (
    consolidate_archive_node,
    save_generation_node,
)
from deepagent.nodes.review import (
    advisor_stage_node,
    exit_node,
    modifier_stage_node,
    review_generation_node,
)
from deepagent.nodes.regen import regenerate_failed_node


def create_deep_evolution_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("load_or_resume", _trace_wrapped_node("load_or_resume", load_or_resume_node))
    workflow.add_node("init_run_memory", _trace_wrapped_node("init_run_memory", init_run_memory_node))
    workflow.add_node("plan_generation", _trace_wrapped_node("plan_generation", plan_generation_node))
    workflow.add_node("synthesis_plan", _trace_wrapped_node("synthesis_plan", synthesis_plan_node))
    workflow.add_node("research_candidates", _trace_wrapped_node("research_candidates", research_candidates_node))
    workflow.add_node("prepare_synthesis_briefs", _trace_wrapped_node("prepare_synthesis_briefs", prepare_synthesis_briefs_node))
    # Slot fan-out nodes (Option A + Phase B): one Send per non-survivor work
    # item, each runs full per-slot synth → repair → postprocess → validate.
    workflow.add_node("slot_dispatch", _trace_wrapped_node("slot_dispatch", slot_dispatch_node))
    workflow.add_node("slot_unit", _trace_wrapped_node("slot_unit", slot_unit_node))
    workflow.add_node("slot_aggregate", _trace_wrapped_node("slot_aggregate", slot_aggregate_node))
    workflow.add_node("ground_and_rescore", _trace_wrapped_node("ground_and_rescore", ground_and_rescore_node))
    workflow.add_node("regenerate_failed", _trace_wrapped_node("regenerate_failed", regenerate_failed_node))
    workflow.add_node("save_generation", _trace_wrapped_node("save_generation", save_generation_node))
    workflow.add_node("consolidate_archive", _trace_wrapped_node("consolidate_archive", consolidate_archive_node))
    workflow.add_node("review_generation", _trace_wrapped_node("review_generation", review_generation_node))
    workflow.add_node("advisor_stage", _trace_wrapped_node("advisor_stage", advisor_stage_node))
    workflow.add_node("modifier_stage", _trace_wrapped_node("modifier_stage", modifier_stage_node))
    workflow.add_node("exit", _trace_wrapped_node("exit", exit_node))
    workflow.add_conditional_edges(
        START,
        _route_entry_phase,
        {
            "load_or_resume": "load_or_resume",
            "init_run_memory": "init_run_memory",
            "plan_generation": "plan_generation",
            "synthesis_plan": "synthesis_plan",
            "research_candidates": "research_candidates",
            "prepare_synthesis_briefs": "prepare_synthesis_briefs",
            # Resume aliases: legacy session_phase values from older runs
            # still resume correctly by mapping to the slot fan-out path.
            "synthesize_candidates": "slot_dispatch",
            "postprocess_candidates": "slot_aggregate",
            "validate_candidates": "slot_aggregate",
            "repair_failed_candidates": "slot_dispatch",
            "slot_dispatch": "slot_dispatch",
            "slot_unit": "slot_unit",
            "slot_aggregate": "slot_aggregate",
            "ground_and_rescore": "ground_and_rescore",
            "regenerate_failed": "regenerate_failed",
            "save_generation": "save_generation",
            "consolidate_archive": "consolidate_archive",
            "review_generation": "review_generation",
            "advisor_stage": "advisor_stage",
            "modifier_stage": "modifier_stage",
            "exit": "exit",
        },
    )
    workflow.add_edge("load_or_resume", "init_run_memory")
    workflow.add_edge("init_run_memory", "plan_generation")
    workflow.add_conditional_edges(
        "plan_generation",
        _route_context_phase,
        {
            "synthesis_plan": "synthesis_plan",
            "research_candidates": "research_candidates",
            "prepare_synthesis_briefs": "prepare_synthesis_briefs",
            "synthesize_candidates": "slot_dispatch",
        },
    )
    workflow.add_conditional_edges(
        "synthesis_plan",
        _route_context_phase,
        {
            "research_candidates": "research_candidates",
            "prepare_synthesis_briefs": "prepare_synthesis_briefs",
            "synthesize_candidates": "slot_dispatch",
        },
    )
    workflow.add_conditional_edges(
        "research_candidates",
        _route_context_phase,
        {
            "prepare_synthesis_briefs": "prepare_synthesis_briefs",
            "synthesize_candidates": "slot_dispatch",
        },
    )
    workflow.add_edge("prepare_synthesis_briefs", "slot_dispatch")
    # Slot fan-out: dispatcher returns either a list of Send("slot_unit", ...)
    # for parallel per-slot pipelines, or "slot_aggregate" directly when
    # only survivor work_items exist (no LLM work to do).
    workflow.add_conditional_edges(
        "slot_dispatch",
        slot_dispatch_route,
        ["slot_unit", "slot_aggregate"],
    )
    workflow.add_edge("slot_unit", "slot_aggregate")
    workflow.add_edge("slot_aggregate", "save_generation")
    # Phase G: slot_unit handles per-slot ground_and_rescore inline (after
    # validate pass). slot_aggregate routes directly to save_generation.
    # The legacy `ground_and_rescore` node remains registered for backwards
    # compatibility but is unreachable from the slot fan-out path.
    workflow.add_conditional_edges(
        "ground_and_rescore",
        _route_grounding_phase,
        {"save_generation": "save_generation", "regenerate_failed": "regenerate_failed"},
    )
    workflow.add_conditional_edges(
        "regenerate_failed",
        # Two regen exit paths (slot_aggregate is the all-giveup short-circuit
        # target — handled via the START conditional via "slot_aggregate" alias):
        #   "synthesis_plan"        — research_session: items go through
        #                             planning before slot_dispatch.
        #   "synthesize_candidates" — repair-only path: items go directly to
        #                             slot_dispatch where slot_unit detects
        #                             `repair_strategy` and dispatches to
        #                             _run_repair_item (Phase A).
        #   "slot_aggregate"        — all-giveup short-circuit: skip to
        #                             aggregator so elite_backfill fires.
        lambda state: state.get("session_phase") if state.get("session_phase") in {
            "synthesis_plan", "synthesize_candidates", "slot_aggregate",
        } else "synthesize_candidates",
        {
            "synthesis_plan": "synthesis_plan",
            "synthesize_candidates": "slot_dispatch",
            "slot_aggregate": "slot_aggregate",
        },
    )
    workflow.add_edge("save_generation", "consolidate_archive")
    workflow.add_edge("consolidate_archive", "review_generation")
    workflow.add_conditional_edges(
        "review_generation",
        _continue_or_end,
        {
            "init_run_memory": "init_run_memory",
            "advisor_stage": "advisor_stage",
            "modifier_stage": "modifier_stage",
            "exit": "exit",
            END: END,
        },
    )
    workflow.add_edge("advisor_stage", "review_generation")
    workflow.add_edge("modifier_stage", "review_generation")
    workflow.add_edge("exit", END)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
