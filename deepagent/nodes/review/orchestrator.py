"""Review-phase nodes: review_generation, advisor, modifier, exit."""

from typing import Dict

from langchain_core.messages import SystemMessage

from deepagent.nodes.review.advisor import advise_on_generation
from deepagent.nodes.review.modifier import modify_problem
from deepagent.state_full import AgentState
from deepagent.nodes.validation.worker import validate_problem

from deepagent.graph.runtime import (
    _build_review_prompt,
    _completion_stop_reason,
    _cumulative_validated_generated_count,
    _target_problem_count,
)
from deepagent.graph.artifacts import _write_json_artifact


def _parse_modify_payload(payload: Dict) -> Dict:
    problem_id = (payload or {}).get("problem_id", "").strip()
    request = (payload or {}).get("request", "").strip()
    return {"problem_id": problem_id, "request": request}


def review_generation_node(state: AgentState):
    action = (state.get("review_action") or "").strip().lower()
    options = state.get("session_options", {}) or {}
    params = state.get("parameters", {}) or {}
    auto_continue = bool(params.get("auto_continue_generations", True))
    interactive_review = bool(options.get("interactive_review"))
    stop_reason = _completion_stop_reason(state)
    stop_message = ""
    if stop_reason == "max_generations":
        stop_message = f"Reached max generations ({options.get('max_generations')}). Exiting."
    elif stop_reason == "target_problem_count":
        stop_message = (
            f"Reached target problem count ({_cumulative_validated_generated_count(state)}/"
            f"{_target_problem_count(state)}). Exiting."
        )

    if not action:
        if auto_continue and not stop_reason:
            return {
                "session_phase": "generation_handoff_ready",
                "awaiting_review": False,
                "generation_handoff_ready": True,
                "review_action": "",
                "review_payload": {},
                "stop_reason": "",
                "messages": [SystemMessage(content="Auto-continuing to next generation.")],
            }
        if stop_reason and not interactive_review:
            return {
                "session_phase": "exit",
                "awaiting_review": False,
                "generation_handoff_ready": False,
                "stop_reason": stop_reason,
                "messages": [SystemMessage(content=stop_message)],
            }
        result = {
            "session_phase": "generation_review_wait",
            "awaiting_review": True,
            "generation_handoff_ready": False,
            "review_prompt": _build_review_prompt(state),
            "stop_reason": stop_reason,
            "messages": [SystemMessage(content=_build_review_prompt(state))],
        }
        _write_json_artifact(
            {**state, **result},
            "review_generation",
            {
                "generation_count": state.get("generation_count", 0),
                "review_prompt": result["review_prompt"],
                "current_generation": state.get("current_generation", []),
            },
        )
        return result

    if action == "continue":
        if stop_reason:
            return {
                "session_phase": "exit",
                "awaiting_review": False,
                "generation_handoff_ready": False,
                "stop_reason": stop_reason,
                "messages": [SystemMessage(content=stop_message)],
            }
        return {
            "session_phase": "generation_handoff_ready",
            "awaiting_review": False,
            "generation_handoff_ready": True,
            "review_action": "",
            "review_payload": {},
            "stop_reason": "",
            "messages": [SystemMessage(content="Continuing to next generation.")],
        }
    if action == "advisor":
        return {"session_phase": "advisor_stage", "awaiting_review": False, "generation_handoff_ready": False, "stop_reason": stop_reason}
    if action == "modify":
        return {"session_phase": "modifier_stage", "awaiting_review": False, "generation_handoff_ready": False, "stop_reason": stop_reason}
    if action == "exit":
        return {"session_phase": "exit", "awaiting_review": False, "generation_handoff_ready": False, "stop_reason": stop_reason or "manual_exit"}
    return {
        "session_phase": "generation_review_wait",
        "awaiting_review": True,
        "generation_handoff_ready": False,
        "review_prompt": _build_review_prompt(state),
        "stop_reason": stop_reason,
        "messages": [SystemMessage(content="Unknown review action. Expected continue, advisor, modify, or exit.")],
    }


def advisor_stage_node(state: AgentState):
    payload = state.get("review_payload", {}) or {}
    request = payload.get("request") or "Analyze generation and suggest next-step parameter updates."
    result = advise_on_generation(
        state.get("current_generation", []) or [],
        request,
        archival_memory_handle=state.get("archival_memory_handle", {}) or {},
        run_working_memory=state.get("run_working_memory", {}) or {},
    )
    params = dict(state.get("parameters", {}) or {})
    params.update(result.get("parameter_updates", {}) or {})
    return {
        "advisor_result": result,
        "parameters": params,
        "session_phase": "review_generation",
        "review_action": "",
        "review_payload": {},
        "messages": [SystemMessage(content=result.get("advice", "No advice"))],
    }


def modifier_stage_node(state: AgentState):
    payload = _parse_modify_payload(state.get("review_payload", {}) or {})
    current_generation = list(state.get("current_generation", []) or [])
    if not payload["problem_id"] or not payload["request"]:
        return {
            "modifier_result": {"status": "invalid_request"},
            "session_phase": "review_generation",
            "review_action": "",
            "review_payload": {},
            "messages": [SystemMessage(content="Modify action requires problem_id and request.")],
        }

    index_map = {problem.get("id"): idx for idx, problem in enumerate(current_generation)}
    if payload["problem_id"] not in index_map:
        return {
            "modifier_result": {"status": "unknown_problem", "problem_id": payload["problem_id"]},
            "session_phase": "review_generation",
            "review_action": "",
            "review_payload": {},
            "messages": [SystemMessage(content=f"Unknown problem id: {payload['problem_id']}")],
        }

    original = current_generation[index_map[payload["problem_id"]]]
    modified = modify_problem(original, payload["request"])
    if not modified:
        return {
            "modifier_result": {"status": "failed", "problem_id": payload["problem_id"]},
            "session_phase": "review_generation",
            "review_action": "",
            "review_payload": {},
            "messages": [SystemMessage(content=f"Modification failed for {payload['problem_id']}")],
        }

    verdict, reason, validation_assessment = validate_problem(modified)
    modified["validation_assessment"] = validation_assessment
    if verdict == "hard_fail":
        return {
            "modifier_result": {"status": "rejected", "problem_id": payload["problem_id"], "reason": reason},
            "session_phase": "review_generation",
            "review_action": "",
            "review_payload": {},
            "messages": [SystemMessage(content=f"Modified problem rejected: {reason}")],
        }

    current_generation[index_map[payload["problem_id"]]] = modified
    return {
        "current_generation": current_generation,
        "modifier_result": {"status": "applied", "problem_id": payload["problem_id"]},
        "session_phase": "review_generation",
        "review_action": "",
        "review_payload": {},
        "messages": [SystemMessage(content=f"Applied modification to {payload['problem_id']}: {reason}")],
    }


def exit_node(state: AgentState):
    return {
        "session_phase": "exit",
        "awaiting_review": False,
        "should_exit": True,
        "messages": [SystemMessage(content="Exiting session.")],
    }
