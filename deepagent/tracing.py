"""Unified tracing, dispatch, and parallel-execution infrastructure."""

import hashlib
import json
import logging
import time
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import ContextVar
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langsmith.run_helpers import get_current_run_tree, tracing_context
from langsmith.run_trees import RunTree

from deepagent.state_full import AgentState


logger = logging.getLogger("deep_tracing")

# Retryable structured-output errors: LLMs occasionally truncate JSON or stop
# mid-response. Without a retry boundary, these transient failures escape
# the graph and kill multi-generation runs (~100% fatality over 10 gens).
# Single retry at this shared invocation point covers researcher,
# plan_generation, synthesis_plan, validator, regen_planner, etc.
try:
    from pydantic import ValidationError as _PydanticValidationError
    from pydantic_core import ValidationError as _PydanticCoreValidationError
except ImportError:
    _PydanticValidationError = None
    _PydanticCoreValidationError = None
try:
    from openai import LengthFinishReasonError as _LengthFinishReasonError
except ImportError:
    _LengthFinishReasonError = None
try:
    from langchain_core.exceptions import OutputParserException as _OutputParserException
except ImportError:
    _OutputParserException = None

_RETRYABLE_STRUCTURED_ERRORS = tuple(
    cls for cls in (
        _PydanticValidationError,
        _PydanticCoreValidationError,
        _LengthFinishReasonError,
        _OutputParserException,
    )
    if cls is not None
)


def _is_retryable_structured_error(exc: BaseException) -> bool:
    """Treat transient structured-output failures as retryable."""
    if _RETRYABLE_STRUCTURED_ERRORS and isinstance(exc, _RETRYABLE_STRUCTURED_ERRORS):
        return True
    msg = str(exc).lower()
    if "invalid json" in msg or "eof while parsing" in msg:
        return True
    if "finish_reason" in msg and "length" in msg:
        return True
    return False


# ── Message & envelope helpers ──────────────────────────────────────────


def _new_tool_call_id(prefix: str = "tc") -> str:
    """Generate a short tool_call_id for orchestrator-synthesized dispatches."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def build_dispatch_tool_call(name: str, args: Dict[str, Any], tool_call_id: Optional[str] = None) -> Dict[str, Any]:
    """Build a single tool_call payload for an AIMessage.

    ``args`` should already be validated against a pydantic schema before being
    passed in; the helper only formats the envelope.
    """
    return {
        "name": name,
        "args": dict(args or {}),
        "id": tool_call_id or _new_tool_call_id(),
        "type": "tool_call",
    }


def build_dispatch_envelope(tool_calls: List[Dict[str, Any]], content: str = "") -> AIMessage:
    """Build an AIMessage carrying the dispatch tool_calls.

    The orchestrator uses this to record "I dispatched these slots to workers"
    in state.messages. Appears in LangSmith as a normal tool-using AIMessage.
    """
    return AIMessage(content=content, tool_calls=list(tool_calls or []))


def wrap_dispatch_result(
    tool_call_id: str,
    name: str,
    result: Any,
    *,
    status: str = "success",
) -> ToolMessage:
    """Wrap a worker result as a ToolMessage pointing back at the tool_call."""
    if isinstance(result, (dict, list)):
        content = json.dumps(result, ensure_ascii=False)
    else:
        content = str(result)
    message = ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=name,
        status=status,
    )
    return message


def digest_payload(payload: Any, length: int = 12) -> str:
    """Short deterministic digest for dispatch args integrity checks."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def serialize_message(message: BaseMessage) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": message.__class__.__name__,
        "content": message.content if isinstance(message.content, str) else str(message.content),
    }
    if isinstance(message, ToolMessage):
        payload["name"] = getattr(message, "name", "")
        payload["tool_call_id"] = getattr(message, "tool_call_id", "")
    elif isinstance(message, AIMessage):
        payload["tool_calls"] = getattr(message, "tool_calls", []) or []
        payload["invalid_tool_calls"] = getattr(message, "invalid_tool_calls", []) or []
    elif isinstance(message, (SystemMessage, HumanMessage)):
        payload["name"] = getattr(message, "name", "")
    return payload


def serialize_messages(messages: Iterable[BaseMessage]) -> List[Dict[str, Any]]:
    return [serialize_message(message) for message in messages or []]


def slim_token_usage(response_metadata: Dict[str, Any]) -> Dict[str, Any]:
    token_usage = dict((response_metadata or {}).get("token_usage", {}) or {})
    return {
        "prompt_tokens": token_usage.get("prompt_tokens", 0),
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
        "cost": token_usage.get("cost"),
    }


# ── LLM invocation wrapper ─────────────────────────────────────────────


def invoke_structured_with_slim_trace(
    structured_llm,
    messages: List[BaseMessage],
    *,
    invoke_config: Optional[Dict[str, Any]] = None,
    trace_name: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    summary_inputs: Optional[Dict[str, Any]] = None,
    output_key: str = "result",
    max_structured_retries: int = 2,
) -> Dict[str, Any]:
    """Structured LLM invocation with explicit slim chain wrapper.

    The wrapper is REQUIRED in this codebase to establish the parent-child
    relationship: langchain's auto-trace mechanism does not propagate the
    manually-pushed parent (`_MANUAL_TRACE_PARENT`) used by the rest of
    the tracing infrastructure. Without the explicit
    `parent.create_child(...)` wrapper here, LLM auto-traces float as
    separate top-level runs in LangSmith UI.

    The wrapper:
      1. Resolves the ambient parent via `get_current_run_tree()` (which
         is set by `_trace_wrapped_node` for orchestrator nodes and by
         `tracing_context(parent=slot_run)` for slot fan-out branches).
      2. Creates a single chain span named `trace_name` as a child of that
         parent.
      3. Suppresses LangChain's own auto-trace (via
         `tracing_context(enabled=False)`) to avoid duplicate sibling spans.
      4. Records LLM output, model name, finish reason, and slimmed token
         usage on the wrapper.
    """
    parent = get_current_run_tree()
    slim_run = None
    if parent is not None:
        slim_run = parent.create_child(
            name=trace_name,
            run_type="chain",
            inputs={
                "summary": dict(summary_inputs or {}),
                "messages": serialize_messages(messages),
            },
            tags=list(tags or []),
            extra={"metadata": dict(metadata or {})},
        )
        slim_run.post()

    last_exc: Optional[BaseException] = None
    retry_events: List[str] = []
    try:
        for attempt in range(max_structured_retries + 1):
            try:
                with tracing_context(enabled=False):
                    response = structured_llm.invoke(messages, config=invoke_config)
                if response.get("parsing_error") is not None:
                    raise response["parsing_error"]
                parsed = response["parsed"]
                parsed_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
                raw_message = response["raw"]
                response_metadata = getattr(raw_message, "response_metadata", {}) or {}
                if slim_run is not None:
                    slim_run.end(
                        outputs={
                            output_key: parsed_dict,
                            "output_text": getattr(raw_message, "content", ""),
                            "model_name": response_metadata.get("model_name", ""),
                            "finish_reason": response_metadata.get("finish_reason", ""),
                            "token_usage": slim_token_usage(response_metadata),
                            "structured_retries": attempt,
                            "retry_events": retry_events,
                        }
                    )
                return parsed_dict
            except Exception as exc:
                if attempt < max_structured_retries and _is_retryable_structured_error(exc):
                    retry_events.append(f"attempt_{attempt + 1}:{type(exc).__name__}")
                    logger.warning(
                        "%s: structured-output attempt %d/%d failed with retryable %s; retrying.",
                        trace_name, attempt + 1, max_structured_retries + 1, type(exc).__name__,
                    )
                    time.sleep(0.5 * (attempt + 1))  # small backoff
                    continue
                last_exc = exc
                raise
    except Exception as exc:
        if slim_run is not None:
            slim_run.end(
                outputs={
                    "status": "error",
                    "structured_retries": len(retry_events),
                    "retry_events": retry_events,
                },
                error=f"{type(exc).__name__}: {exc}",
            )
        raise
    finally:
        if slim_run is not None:
            slim_run.patch()


# ── Deterministic span recording ───────────────────────────────────────


def record_deterministic_span(
    name: str,
    *,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_type: str = "tool",
    parent_run: Optional[Any] = None,
) -> None:
    """Record a synchronous deterministic operation as a LangSmith child run.

    Use this to surface Python-only computations (retrievals, guards, checks)
    in LangSmith traces. Without it, only LLM calls appear — deterministic
    operations are invisible, making it hard to see what was queried vs. what
    matched. Example call sites:
      - `retrieve_topk_archive_by_text` (grounding gate novelty retrieval)
      - `assess_near_copy` (validator guard)
      - `_enforce_survivor_answer_guard`, `_enforce_fixed_point_depth`
        (selector post-plan guards)

    `inputs` should describe WHAT was queried (candidate id, parent ids, etc.);
    `outputs` should describe WHAT was checked + the decision (matches,
    similarity scores, verdict, swap target). Keep both compact.

    No-op when not inside a LangSmith trace (parent_run is None and there is
    no ambient run tree).
    """
    parent = parent_run if parent_run is not None else get_current_run_tree()
    if parent is None:
        return
    child = parent.create_child(
        name=name,
        run_type=run_type,
        inputs=dict(inputs or {}),
        tags=list(tags or []),
        extra={"metadata": dict(metadata or {})},
    )
    child.post()
    child.end(outputs=dict(outputs or {}))
    child.patch()


# ── Trace context management ───────────────────────────────────────────


_MANUAL_TRACE_PARENT: ContextVar[Optional[RunTree]] = ContextVar("deepagent_manual_trace_parent", default=None)


def push_manual_trace_parent(parent: Optional[RunTree]):
    return _MANUAL_TRACE_PARENT.set(parent)


def pop_manual_trace_parent(token) -> None:
    _MANUAL_TRACE_PARENT.reset(token)


def _resolve_trace_parent() -> Optional[RunTree]:
    return get_current_run_tree() or _MANUAL_TRACE_PARENT.get()


_SLOT_TRACE_REGISTRY: Dict[str, Dict[str, Any]] = {}
_SLOT_TRACE_LOCK = Lock()


def _active_generation_number(state: Optional[AgentState]) -> int:
    if not state:
        return 0
    base = int(state.get("generation_count", 0) or 0)
    phase = str(state.get("session_phase", "") or "")
    return base if phase in {"consolidate_archive", "review_generation", "advisor_stage", "modifier_stage", "exit"} else base + 1


def _trace_item_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    problem = dict(item.get("problem") or {}) if isinstance(item, dict) else {}
    slot = item.get("_slot", item.get("slot", problem.get("_slot", problem.get("slot", 0))))
    pair_id = item.get("pair_id", problem.get("pair_id"))
    op_type = item.get("op_type", problem.get("op_type", problem.get("type", "")))
    parent_ids = item.get("parent_ids", problem.get("parent_ids"))
    problem_id = problem.get("id", item.get("id", ""))
    return {
        "slot": slot,
        "pair_id": pair_id,
        "op_type": op_type,
        "parent_ids": list(parent_ids or []),
        "problem_id": problem_id,
    }


def _slot_trace_registry_key(state: Optional[AgentState], slot: Any) -> str:
    session_thread_id = ""
    generation_number = 0
    if state:
        session_thread_id = str(state.get("session_thread_id", "") or "")
        generation_number = _active_generation_number(state)
    return f"{session_thread_id}:{generation_number}:{slot}"


def ensure_slot_trace_parent(
    state: Optional[AgentState],
    item: Dict[str, Any],
    *,
    parent_run: Optional[RunTree] = None,
) -> Optional[RunTree]:
    fields = _trace_item_fields(item)
    slot = fields["slot"]
    if slot is None:
        return parent_run if parent_run is not None else _resolve_trace_parent()
    key = _slot_trace_registry_key(state, slot)
    with _SLOT_TRACE_LOCK:
        existing = _SLOT_TRACE_REGISTRY.get(key)
        if existing is not None:
            return existing["run"]
        parent = parent_run if parent_run is not None else _resolve_trace_parent()
        if parent is None:
            return None
        root_name = str(item.get("_slot_trace_name") or f"generator.operation.slot_{slot}")
        tags = list(item.get("_slot_trace_tags") or ["deepagent", "generator", "operation"])
        metadata = {
            "trace_name": root_name,
            "slot": slot,
            "pair_id": fields["pair_id"],
            "op_type": fields["op_type"],
            "problem_id": fields["problem_id"],
            "generation_number": _active_generation_number(state),
        }
        metadata.update(dict(item.get("_slot_trace_metadata") or {}))
        child = parent.create_child(
            name=root_name,
            run_type="chain",
            inputs={
                "slot": slot,
                "pair_id": fields["pair_id"],
                "op_type": fields["op_type"],
                "parent_ids": fields["parent_ids"],
            },
            tags=tags,
            extra={"metadata": metadata},
        )
        child.post()
        _SLOT_TRACE_REGISTRY[key] = {
            "run": child,
            "slot": slot,
            "generation_number": _active_generation_number(state),
            "session_thread_id": str((state or {}).get("session_thread_id", "") or ""),
        }
        return child


def lookup_slot_trace_run(state: Optional[AgentState], slot: int):
    """Look up the active slot trace span (if any) for the given slot.

    Phase E: lets cross-slot orchestrator nodes (e.g. `regenerate_failed_node`)
    create per-slot child spans (like `regen_plan.slot_X`) under the slot's
    own `generator.operation.slot_X` parent. Returns None if the slot has
    no active span (e.g. for the initial dispatch before slot_unit runs).
    """
    key = _slot_trace_registry_key(state, slot)
    with _SLOT_TRACE_LOCK:
        payload = _SLOT_TRACE_REGISTRY.get(key)
    return payload.get("run") if payload else None


def close_one_slot_trace(
    state: Optional[AgentState],
    slot: int,
    *,
    output: Optional[Dict[str, Any]] = None,
    status: str = "completed",
) -> None:
    """Close a single slot's trace span and remove it from the registry.

    Used by `slot_unit_node` after each per-slot pipeline invocation so the
    slot's trace span only spans the active work — NOT the fan-in barrier
    wait or the cross-slot orchestration (slot_aggregate, regenerate_failed,
    regen_planner) that runs between attempts. The next slot_unit
    invocation creates a fresh span via `ensure_slot_trace_parent`, so a
    multi-attempt slot appears as multiple discrete spans in LangSmith UI
    with no internal "empty time" gaps.
    """
    key = _slot_trace_registry_key(state, slot)
    with _SLOT_TRACE_LOCK:
        payload = _SLOT_TRACE_REGISTRY.pop(key, None)
    if payload is None:
        return
    run = payload.get("run")
    if run is None:
        return
    outputs = dict(output or {})
    outputs.setdefault("status", status)
    try:
        run.end(outputs=outputs)
        run.patch()
    except Exception:  # noqa: BLE001 — never let trace cleanup raise
        pass


def close_slot_trace_parents(
    state: Optional[AgentState],
    *,
    slot_outputs: Optional[Dict[int, Dict[str, Any]]] = None,
    default_status: str = "completed",
) -> None:
    generation_number = _active_generation_number(state)
    session_thread_id = str((state or {}).get("session_thread_id", "") or "")
    to_close: List[Tuple[str, Dict[str, Any]]] = []
    with _SLOT_TRACE_LOCK:
        for key, payload in list(_SLOT_TRACE_REGISTRY.items()):
            if payload.get("generation_number") != generation_number:
                continue
            if session_thread_id and payload.get("session_thread_id") != session_thread_id:
                continue
            to_close.append((key, payload))
            del _SLOT_TRACE_REGISTRY[key]
    for _key, payload in to_close:
        run = payload.get("run")
        slot = int(payload.get("slot", 0) or 0)
        outputs = dict((slot_outputs or {}).get(slot) or {})
        outputs.setdefault("status", default_status)
        run.end(outputs=outputs)
        run.patch()


# ── Node-level tracing ─────────────────────────────────────────────────


_TRACE_STAGE_DISPLAY = {
    "load_or_resume": "orchestrator.load_or_resume",
    "init_run_memory": "orchestrator.init_run_memory",
    "plan_generation": "orchestrator.plan_generation.controller",
    "synthesis_plan": "orchestrator.synthesis_plan.controller",
    "research_candidates": "orchestrator.research_candidates",
    "prepare_synthesis_briefs": "orchestrator.brief_validator",
    # Slot fan-out nodes — replaced legacy synthesize / postprocess /
    # repair_failed / validate single-phase nodes (Phase A/B/C).
    "slot_dispatch": "orchestrator.slot_dispatch",
    "slot_unit": "orchestrator.slot_unit",
    "slot_aggregate": "orchestrator.slot_aggregate",
    "ground_and_rescore": "orchestrator.ground_and_rescore",
    "regenerate_failed": "orchestrator.regenerate_failed",
    "save_generation": "orchestrator.save_generation",
    "consolidate_archive": "orchestrator.consolidate_archive",
    "review_generation": "orchestrator.review_generation",
    "advisor_stage": "orchestrator.advisor_stage",
    "modifier_stage": "orchestrator.modifier_stage",
    "exit": "orchestrator.exit",
}


def _planning_generation_snapshot(state: AgentState) -> Dict:
    current_generation = list(state.get("current_generation", []) or [])
    difficulty_mix = Counter(str(problem.get("difficulty", "unknown")) for problem in current_generation)
    family_mix = Counter(str(problem.get("family_signature", "unknown")) for problem in current_generation)
    type_mix = Counter(str(problem.get("type", "unknown")) for problem in current_generation)
    return {
        "current_generation_size": len(current_generation),
        "generation_count": state.get("generation_count", 0),
        "run_working_memory_slots": len(((state.get("run_working_memory", {}) or {}).get("context_packs", {}) or {})),
        "difficulty_mix": dict(difficulty_mix.most_common(6)),
        "family_mix": dict(family_mix.most_common(6)),
        "type_mix": dict(type_mix.most_common(6)),
    }


def _memory_inputs_snapshot(stage: str, state: AgentState) -> Dict[str, Any]:
    run_memory = dict(state.get("run_working_memory", {}) or {})
    archival = dict(state.get("archival_memory_handle", {}) or {})
    slot_packs = dict(run_memory.get("context_packs", {}) or {})
    slot_digests = {slot: pack.get("digest", "") for slot, pack in slot_packs.items()}
    selector_view = dict((run_memory.get("stage_views", {}) or {}).get("selector", {}) or {})
    payload: Dict[str, Any] = {
        "run_working_memory": {
            "scope": "run_working_memory",
            "slot_count": len(slot_packs),
            "slot_digests": slot_digests,
            "selector_tokens": int(selector_view.get("token_estimate", 0) or 0),
            "summary": run_memory.get("run_summary", ""),
        }
    }
    if stage in {"init_run_memory", "slot_aggregate", "consolidate_archive", "advisor_stage"}:
        payload["archival_memory"] = {
            "scope": "archival_memory",
            "problem_card_count": len(archival.get("problem_cards", []) or []),
            "generation_card_count": len(archival.get("generation_cards", []) or []),
            "source": ((archival.get("metrics", {}) or {}).get("source", "")),
        }
    return payload


def _memory_updates_snapshot(stage: str, result: Dict) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    run_memory = dict(result.get("run_working_memory", {}) or {})
    if run_memory:
        payload["run_working_memory"] = {
            "scope": "run_working_memory",
            "slot_count": len((run_memory.get("context_packs", {}) or {})),
            "accepted_candidate_deltas": len(run_memory.get("accepted_candidate_deltas", []) or []),
            "summary": run_memory.get("run_summary", ""),
        }
    if stage == "slot_aggregate":
        evidence = dict(result.get("validation_evidence", {}) or {})
        payload["validation_evidence"] = {
            "scope": "archival_validation",
            "candidate_count": len(evidence),
            "evidence_digests": {
                problem_id: (pack or {}).get("evidence_digest", "")
                for problem_id, pack in evidence.items()
            },
        }
    if stage == "consolidate_archive":
        archival = dict(result.get("archival_memory_handle", {}) or {})
        payload["archival_memory"] = {
            "scope": "archival_memory",
            "problem_card_count": len(archival.get("problem_cards", []) or []),
            "generation_card_count": len(archival.get("generation_cards", []) or []),
            "paths": dict(archival.get("paths", {}) or {}),
        }
    return payload


def _trace_node_inputs(stage: str, state: AgentState) -> Dict:
    from deepagent.graph.artifacts import _slim_problem  # lazy to avoid circular import
    from deepagent.graph.helpers import _slot_failure_registry  # lazy to avoid circular import
    current_ids = [problem.get("id") for problem in (state.get("current_generation", []) or []) if problem.get("id")]
    base_memory = {"memory_inputs": _memory_inputs_snapshot(stage, state)}
    if stage == "load_or_resume":
        return base_memory
    if stage == "init_run_memory":
        return {
            **base_memory,
            "generation_count": state.get("generation_count", 0),
            "current_generation_size": len(current_ids),
            "load_source": state.get("load_source", ""),
        }
    if stage == "plan_generation":
        return {**_planning_generation_snapshot(state), **base_memory}
    if stage == "synthesis_plan":
        return {
            **base_memory,
            "plan_items": [
                {
                    "slot": item.get("slot", 0),
                    "pair_id": item.get("pair_id"),
                    "op_type": item.get("op_type", ""),
                    "variation_axis": item.get("variation_axis", ""),
                    "seed_focus": item.get("seed_focus", ""),
                }
                for item in (state.get("work_items", []) or [])
                if item.get("op_type") != "survivor"
            ],
        }
    if stage == "research_candidates":
        return {
            **base_memory,
            "research_items": [
                {
                    "slot": item.get("slot", 0),
                    "pair_id": item.get("pair_id"),
                    "plan_query_hint": item.get("query_hint", ""),
                    "plan_seed_focus": item.get("seed_focus", ""),
                    "query_hint": ((item.get("context_pack", {}) or {}).get("research_policy", {}) or {}).get("query_hint", ""),
                    "preferred_sources": ((item.get("context_pack", {}) or {}).get("research_policy", {}) or {}).get("preferred_sources", []),
                }
                for item in (state.get("work_items", []) or [])
                if item.get("op_type") != "survivor"
            ]
        }
    if stage == "prepare_synthesis_briefs":
        return {
            **base_memory,
            "brief_items": [
                {"slot": item.get("slot", 0), "pair_id": item.get("pair_id"), "op_type": item.get("op_type", "")}
                for item in (state.get("work_items", []) or [])
                if item.get("op_type") != "survivor"
            ]
        }
    if stage == "synthesize_candidates":
        return {
            **base_memory,
            "work_items": [
                {
                    "slot": item.get("slot", 0),
                    "op_type": item.get("op_type", ""),
                    "pair_id": item.get("pair_id"),
                    "parent_ids": item.get("parent_ids", []) or [],
                    "execution_group": item.get("execution_group", ""),
                }
                for item in (state.get("work_items", []) or [])
            ]
        }
    if stage == "repair_failed_candidates":
        return {**base_memory, "repair_queue": [
            {
                "slot": item.get("slot", 0),
                "pair_id": item.get("pair_id"),
                "op_type": item.get("op_type", ""),
                "repair_strategy": item.get("repair_strategy", ""),
                "failure_type": item.get("failure_type", ""),
            }
            for item in (state.get("repair_queue", []) or [])
        ]}
    if stage == "postprocess_candidates":
        return {**base_memory, "candidate_ids": [_slim_problem(problem) for problem in (state.get("candidates", []) or [])]}
    if stage == "validate_candidates":
        return {
            **base_memory,
            "candidate_ids": [_slim_problem(problem) for problem in (state.get("candidates", []) or [])],
            "failure_registry": {
                slot: {
                    "attempts": value.get("attempts", 0),
                    "last_failure_type": value.get("last_failure_type", ""),
                    "last_repair_strategy": value.get("last_repair_strategy", ""),
                }
                for slot, value in (_slot_failure_registry(state)).items()
            },
        }
    if stage == "save_generation":
        return {**base_memory, "approved_slots": [item.get("slot", 0) for item in (state.get("approved_candidates", []) or [])]}
    if stage == "review_generation":
        return {
            **base_memory,
            "generation_count": state.get("generation_count", 0),
            "valid_count": len(state.get("current_generation", []) or []),
            "current_generation_size": len(current_ids),
        }
    return {**base_memory, "generation_count": state.get("generation_count", 0)}


def _serialize_dispatch_messages(result: Dict) -> List[Dict]:
    """Flatten AIMessage(tool_calls) + ToolMessage pairs from a node result.

    LangSmith only sees what :func:`_trace_node_outputs` returns; the default
    view trims ``messages`` so the Q4 dispatch envelopes emitted by
    synthesis_plan / research_candidates / synthesize_candidates were invisible.
    This helper exposes them as a compact ``dispatch_transcript`` payload.
    """
    from langchain_core.messages import AIMessage, ToolMessage  # local import to avoid top-level cycles
    transcript: List[Dict] = []
    for msg in result.get("messages", []) or []:
        if isinstance(msg, AIMessage):
            tool_calls = list(getattr(msg, "tool_calls", []) or [])
            if not tool_calls:
                continue
            transcript.append(
                {
                    "type": "AIMessage.tool_calls",
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                    "tool_calls": [
                        {
                            "name": tc.get("name"),
                            "id": tc.get("id"),
                            "args": tc.get("args") or {},
                        }
                        for tc in tool_calls
                    ],
                }
            )
        elif isinstance(msg, ToolMessage):
            transcript.append(
                {
                    "type": "ToolMessage",
                    "name": getattr(msg, "name", ""),
                    "tool_call_id": getattr(msg, "tool_call_id", ""),
                    "status": getattr(msg, "status", "success"),
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                }
            )
    return transcript


def _trace_node_outputs(stage: str, result: Dict) -> Dict:
    from deepagent.graph.artifacts import (  # lazy to avoid circular import
        _slim_problem, _slim_failed_problem, _slim_research_artifact,
        _slim_brief, _slim_work_item,
    )
    base_updates = {"memory_updates": _memory_updates_snapshot(stage, result)}
    if stage == "load_or_resume":
        return {**base_updates, "load_source": result.get("load_source", ""), "next_phase": result.get("session_phase", "")}
    if stage == "init_run_memory":
        bank = result.get("archival_memory_handle", {}) or {}
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "problem_card_count": len(bank.get("problem_cards", []) or []),
            "generation_card_count": len(bank.get("generation_cards", []) or []),
            "memory_metrics": bank.get("metrics", {}),
        }
    if stage == "plan_generation":
        strategy = result.get("selection_strategy", {}) or {}
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "normalization_mode": strategy.get("normalization_mode", ""),
            "desired_generation_size": strategy.get("desired_generation_size", 0),
            "active_pool_ids": strategy.get("active_pool_ids", []),
            "strategy_source": strategy.get("strategy_source", ""),
            "op_type_allocation": strategy.get("op_type_allocation", {}),
            "dispatch_outline": [_slim_work_item(item) for item in (result.get("work_items", []) or [])],
        }
    if stage == "synthesis_plan":
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "synthesis_plans": {
                key: {
                    "slot": plan.get("slot"),
                    "pair_id": plan.get("pair_id"),
                    "op_type": plan.get("op_type"),
                    "preferred_composition_pattern": plan.get("preferred_composition_pattern"),
                    "parameter_reuse_policy": plan.get("parameter_reuse_policy"),
                    "research_focus": plan.get("research_focus", ""),
                }
                for key, plan in (result.get("synthesis_plans", {}) or {}).items()
            },
            "dispatch_transcript": _serialize_dispatch_messages(result),
        }
    if stage == "research_candidates":
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "research_artifacts": [_slim_research_artifact(artifact) for artifact in (result.get("research_artifacts", []) or [])],
            "research_skipped_slots": result.get("research_skipped_slots", []),
            "dispatch_transcript": _serialize_dispatch_messages(result),
        }
    if stage == "prepare_synthesis_briefs":
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "synthesis_briefs": {key: _slim_brief(value) for key, value in (result.get("synthesis_briefs", {}) or {}).items()},
            "llm_brief_skipped_slots": result.get("llm_brief_skipped_slots", []),
            "fast_path_slot_count": result.get("fast_path_slot_count", 0),
        }
    if stage == "synthesize_candidates":
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "candidates": [_slim_problem(problem) for problem in (result.get("candidates", []) or [])],
            "failed_problems": [_slim_failed_problem(problem) for problem in (result.get("failed_problems", []) or [])],
            "dispatch_transcript": _serialize_dispatch_messages(result),
        }
    if stage == "repair_failed_candidates":
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "candidates": [_slim_problem(problem) for problem in (result.get("candidates", []) or [])],
            "failed_problems": [_slim_failed_problem(problem) for problem in (result.get("failed_problems", []) or [])],
        }
    if stage == "postprocess_candidates":
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "candidates": [
                {
                    "id": problem.get("id", ""),
                    "slot": problem.get("_slot", problem.get("slot", 0)),
                    "answer": problem.get("answer", ""),
                    "worker_answer": problem.get("worker_answer", ""),
                    "grounding_status": ((problem.get("grounding_assessment", {}) or {}).get("grounding_status", "")),
                }
                for problem in (result.get("candidates", []) or [])
            ],
            "failed_problems": [_slim_failed_problem(problem) for problem in (result.get("failed_problems", []) or [])],
        }
    if stage == "validate_candidates":
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "valid_count": result.get("valid_count", 0),
            "approved_slots": [item.get("slot", 0) for item in (result.get("approved_candidates", []) or [])],
            "failed_problems": [_slim_failed_problem(problem) for problem in (result.get("failed_problems", []) or [])],
        }
    if stage == "save_generation":
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "generation_count": result.get("generation_count", 0),
            "saved_problem_ids": [problem.get("id") for problem in (result.get("current_generation", []) or [])],
        }
    if stage == "consolidate_archive":
        handle = result.get("archival_memory_handle", {}) or {}
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "archive_metrics": handle.get("metrics", {}),
            "archive_paths": handle.get("paths", {}),
        }
    if stage == "review_generation":
        return {
            **base_updates,
            "next_phase": result.get("session_phase", ""),
            "generation_handoff_ready": result.get("generation_handoff_ready", False),
            "awaiting_review": result.get("awaiting_review", False),
        }
    return {**base_updates, "next_phase": result.get("session_phase", "")}


def _trace_wrapped_node(stage: str, fn):
    def _wrapped(state: AgentState):
        parent = _resolve_trace_parent()
        if parent is None:
            return fn(state)
        trace_name = _TRACE_STAGE_DISPLAY.get(stage, stage)
        child = parent.create_child(
            name=trace_name,
            run_type="chain",
            inputs=_trace_node_inputs(stage, state),
            tags=["deepagent", "node", stage],
            extra={"metadata": {"stage": stage, "trace_name": trace_name, "generation_count": state.get("generation_count", 0)}},
        )
        child.post()
        try:
            with tracing_context(parent=child, enabled=True):
                result = fn(state)
            child.end(outputs=_trace_node_outputs(stage, result))
            return result
        except Exception as exc:
            child.end(outputs={"status": "error", "stage": stage}, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            child.patch()

    return _wrapped


# ── Parallel dispatch ──────────────────────────────────────────────────


def _trace_identity(item: Dict, prefix: str) -> str:
    pair_id = item.get("pair_id")
    slot = item.get("slot", 0)
    if pair_id:
        return f"{prefix}.{pair_id}"
    return f"{prefix}.slot_{slot}"


def _run_with_child_trace(parent_run, name: str, item: Dict, worker_fn):
    if parent_run is None:
        return worker_fn(item)
    child = parent_run.create_child(
        name=name,
        run_type="chain",
        inputs={
            "slot": item.get("slot"),
            "pair_id": item.get("pair_id"),
            "op_type": item.get("op_type"),
            "parent_ids": item.get("parent_ids"),
        },
        tags=["deepagent", "worker"],
        extra={"metadata": {"trace_name": name}},
    )
    child.post()
    try:
        with tracing_context(parent=child, enabled=True):
            result = worker_fn(item)
        child.end(outputs={"status": "completed"})
        return result
    except Exception as exc:
        child.end(outputs={"status": "failed"}, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        child.patch()


def _dispatch_fifo_parallel(items: List[Dict], max_parallel: int, worker_fn, *, slot_trace_state: Optional[AgentState] = None):
    if not items:
        return []
    max_workers = max(1, min(max_parallel, len(items)))
    results = []
    next_index = 0
    future_to_item = {}
    parent_run = get_current_run_tree()

    def _traced_worker(item: Dict):
        trace_name = item.get("_trace_name") or _trace_identity(item, f"deepagent.{item.get('op_type', 'worker')}")
        stage_parent = parent_run
        if slot_trace_state is not None and item.get("_slot_trace_enabled"):
            stage_parent = ensure_slot_trace_parent(slot_trace_state, item, parent_run=parent_run)
        return _run_with_child_trace(stage_parent, trace_name, item, worker_fn)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while next_index < len(items) and len(future_to_item) < max_workers:
            item = items[next_index]
            future_to_item[executor.submit(_traced_worker, item)] = item
            next_index += 1

        while future_to_item:
            done, _ = wait(list(future_to_item.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                item = future_to_item.pop(future)
                results.append((item, future.result()))
                if next_index < len(items):
                    next_item = items[next_index]
                    future_to_item[executor.submit(_traced_worker, next_item)] = next_item
                    next_index += 1
    return results
