"""
DeepAgent full pipeline state (HITL 포함).
LangGraph AgentState for the evolution pipeline: memory bank, context packs,
per-slot candidates, review handoff, and bookkeeping.
"""
import operator
from typing import TypedDict, List, Dict, Annotated
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], operator.add]
    current_generation: List[dict]
    current_generation_size: int
    generation_save_status: str
    generation_count: int
    desired_generation_size: int
    target_generation_size: int
    cumulative_validated_generated_count: int
    stop_reason: str
    session_initialized: bool
    session_phase: str
    session_thread_id: str
    session_options: Dict
    run_metadata: Dict
    selection_strategy: Dict
    work_items: List[dict]
    pair_plans: List[dict]
    pair_results: Dict
    pair_health_scores: Dict
    mutation_policies: Dict
    synthesis_briefs: Dict
    lineage_metrics: Dict
    invariant_bundles: Dict
    run_working_memory: Dict
    archival_memory_handle: Dict
    validation_evidence: Dict
    context_packs: Dict
    research_artifacts: List[dict]
    approved_candidates: List[dict]
    retry_registry: Dict
    slot_retry_registry: Dict
    slot_failure_registry: Dict
    failed_problems: List[dict]
    candidates: List[dict]
    repair_queue: List[dict]
    # Snapshots populated by save_generation_node and consumed by
    # consolidate_archive_node to build plan_outcome_cards. Needed because
    # save_generation clears work_items / failed_problems for the next
    # generation before consolidate can read them.
    last_work_items: List[dict]
    last_failed_problems: List[dict]
    last_saved_slot_map: Dict
    synthesis_plans: Dict
    # Stable copy of the planned work_items captured once at synthesis_plan time.
    # regenerate_failed_node clears state["work_items"] during retry loops, so we
    # keep this separate snapshot alive until consolidate_archive runs and then
    # reset it at the next init_run_memory.
    planned_work_items: List[dict]
    valid_count: int
    validation_feedback: List[str]
    parameters: Dict
    review_action: str
    review_payload: Dict
    awaiting_review: bool
    generation_handoff_ready: bool
    review_prompt: str
    advisor_result: Dict
    modifier_result: Dict
    modification_requests: Dict
    problems_to_modify: List[dict]
    # Ring buffer of the most recent generations' saved-survivor answers.
    # Used by the selector guard (P3.2) to block same-answer survivor chains.
    # Entries are appended in save_generation_node and trimmed to the last 3.
    recent_survivor_answers: List[str]
    # Optional per-candidate grounding + difficulty-rescore result, produced by
    # ground_and_rescore_node (P4) between validate_candidates and save_generation.
    grounding_assessments: Dict
    # Slot fan-out scratch buffer (Option A — LangGraph Send pattern).
    # Each `slot_unit` Send invocation appends a single entry tagged with the
    # generation it belongs to. The aggregator (`slot_aggregate_node`) filters
    # by current generation_count, projects into `candidates` / `failed_problems`,
    # then leaves the buffer in place (entries from prior generations remain
    # but are ignored — small constant memory cost, no clear-needed-mid-run).
    slot_outputs: Annotated[List[Dict], operator.add]
