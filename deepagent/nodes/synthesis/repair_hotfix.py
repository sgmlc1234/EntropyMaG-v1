import json
import logging
import re
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError

from config import get_llm_config
from deepagent.python_sandbox import detect_runtime_mode
from deepagent.tracing import invoke_structured_with_slim_trace
from prompts import (
    CODE_ONLY_HOTFIX_SYSTEM_PROMPT,
    STATEMENT_DOMAIN_HOTFIX_SYSTEM_PROMPT,
    TARGET_ONLY_HOTFIX_SYSTEM_PROMPT,
    CodeOnlyHotfixSchema,
    OrchestratorDispatchSchema,
    OrchestratorPlanSchema,
    StatementDomainHotfixSchema,
    TargetOnlyHotfixSchema,
    build_code_only_hotfix_prompt,
    build_invariant_bundle_block,
    build_orchestrator_context_block,
    build_statement_domain_hotfix_prompt,
    build_target_only_hotfix_prompt,
)

logger = logging.getLogger("repair_hotfix")

_cfg = get_llm_config("hotfix")
_repair_llm = ChatOpenAI(
    model=_cfg["model"],
    temperature=0.0,
    max_tokens=_cfg.get("max_tokens", 16000),
    timeout=_cfg.get("timeout", 120),
    max_retries=_cfg.get("max_retries", 3),
    api_key=_cfg["api_key"],
    base_url=_cfg["base_url"],
    extra_body=_cfg.get("extra_body"),
)


def _brief_enforcement_block(synthesis_brief: Dict) -> str:
    brief = dict(synthesis_brief or {})
    if not brief:
        return ""
    lines = ["Authoritative synthesis brief constraints:"]
    for key in (
        "target_quantity_guard",
        "relation_guard",
        "preferred_composition_pattern",
        "parameter_reuse_policy",
        "deep_variant_requirement",
        "retry_focus",
    ):
        value = str(brief.get(key, "") or "").strip()
        if value:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _combined_hotfix_feedback(validation_feedback: str, synthesis_brief: Dict) -> str:
    parts = [str(validation_feedback or "").strip()]
    brief_block = _brief_enforcement_block(synthesis_brief)
    if brief_block:
        parts.append(brief_block)
    return "\n\n".join(part for part in parts if part)


def _extract_required_replacements(synthesis_brief: Dict) -> List[Dict[str, str]]:
    text = " ".join(
        str((synthesis_brief or {}).get(key, "") or "")
        for key in ("parameter_reuse_policy", "retry_focus")
    )
    replacements = []
    for match in re.finditer(r"replace\s+\$?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+)\$?\s+with(?:\s+a\s+value)?\s+\$?[A-Za-z_][A-Za-z0-9_]*\s*=\s*([0-9]+)\$?", text, re.IGNORECASE):
        replacements.append({"symbol": match.group(1), "old": match.group(2), "new": match.group(3)})
    return replacements


def _enforce_hotfix_brief(candidate_before: Dict, repaired: Dict, synthesis_brief: Dict) -> None:
    brief = dict(synthesis_brief or {})
    if not brief:
        return
    retry_focus = str(brief.get("retry_focus", "") or "")
    parameter_reuse_policy = str(brief.get("parameter_reuse_policy", "") or "")
    timeout_sensitive = "timeout" in retry_focus.lower() or "timeout" in parameter_reuse_policy.lower()
    replacements = _extract_required_replacements(brief)
    code = str(repaired.get("code", "") or "")
    statement = str(repaired.get("statement", candidate_before.get("statement", "")) or "")
    solution = str(repaired.get("solution", "") or "")
    combined = "\n".join([code, statement, solution])

    for item in replacements:
        symbol = item["symbol"]
        old = item["old"]
        new = item["new"]
        old_assign = re.compile(rf"\b{re.escape(symbol)}\s*=\s*{re.escape(old)}\b")
        new_assign = re.compile(rf"\b{re.escape(symbol)}\s*=\s*{re.escape(new)}\b")
        if old_assign.search(code):
            raise RuntimeError(
                f"hotfix_brief_violation: parameter_reuse_policy requires replacing {symbol}={old}, but repaired code still contains it."
            )
        if timeout_sensitive and not new_assign.search(combined):
            raise RuntimeError(
                f"hotfix_brief_violation: timeout retry requires reflecting {symbol}={new}, but repaired output did not incorporate it."
            )

    if timeout_sensitive and code.strip() == str(candidate_before.get("code", "") or "").strip():
        raise RuntimeError(
            "hotfix_brief_violation: timeout retry requires changing the code path, but code_only_hotfix returned the original code unchanged."
        )


def _problem_summary(candidate: Dict) -> str:
    payload = {
        "id": candidate.get("id", ""),
        "statement": candidate.get("statement", ""),
        "answer": str(candidate.get("answer", "")),
        "solution": candidate.get("solution", "") or candidate.get("solution_sketch", ""),
        "code": candidate.get("code", ""),
        "code_runtime_mode": candidate.get("code_runtime_mode", ""),
        "variation_axis_used": candidate.get("variation_axis_used", ""),
        "evidence_summary": candidate.get("evidence_summary", ""),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_context(
    *,
    slot: int,
    op_type: str,
    parent_ids: List[str],
    invariant_bundle_block: str,
    objective: str,
    strategy_summary: str,
    finish_when: List[str],
) -> str:
    plan_payload = OrchestratorPlanSchema(
        stage="repair_failed_candidates",
        objective=objective,
        strategy_summary=strategy_summary,
        hard_constraints=[
            "Treat the orchestrator context as authoritative.",
            "Return JSON only.",
        ],
        success_criteria=finish_when,
    ).model_dump()
    dispatch_payload = OrchestratorDispatchSchema(
        agent_role="repair_hotfix",
        slot=slot,
        op_type=op_type,
        parent_ids=parent_ids,
        invariant_notes="Preserve the exact parent invariant bundle and target quantity semantics.",
        task_payload={},
        finish_when=finish_when,
    ).model_dump()
    return build_orchestrator_context_block(plan_payload, dispatch_payload) + "\n\n" + invariant_bundle_block


def apply_code_only_hotfix(
    candidate: Dict,
    *,
    invariant_bundles: List[Dict] = None,
    synthesis_brief: Dict = None,
    validation_feedback: str = "",
    escalation_signal: str = "",
    invoke_config: Dict = None,
) -> Dict:
    invariant_bundle_block = build_invariant_bundle_block(invariant_bundles or [])
    prompt = build_code_only_hotfix_prompt(
        problem_summary=_problem_summary(candidate),
        invariant_bundle_block=invariant_bundle_block,
        validation_feedback=_combined_hotfix_feedback(validation_feedback, synthesis_brief),
        escalation_signal=escalation_signal,
    )
    context = _build_context(
        slot=int(candidate.get("_slot", candidate.get("slot", 0)) or 0),
        op_type=str(candidate.get("op_type", "mutation") or "mutation"),
        parent_ids=list(candidate.get("parent_ids", []) or []),
        invariant_bundle_block=invariant_bundle_block,
        objective="Repair only the answer/solution/code surface while keeping the statement unchanged.",
        strategy_summary="Use a narrow code-only hotfix instead of full regeneration.",
        finish_when=[
            "Keep the statement unchanged.",
            "Return valid code-only repair fields.",
            "Keep the exact target semantics unchanged.",
        ],
    )
    structured_llm = _repair_llm.with_structured_output(CodeOnlyHotfixSchema, include_raw=True)
    try:
        result = invoke_structured_with_slim_trace(
            structured_llm,
            [
                SystemMessage(content=CODE_ONLY_HOTFIX_SYSTEM_PROMPT),
                HumanMessage(content=context + "\n\n" + prompt),
            ],
            invoke_config=invoke_config,
            trace_name=((invoke_config or {}).get("run_name") or "deepagent.repair_hotfix.code"),
            tags=list((invoke_config or {}).get("tags", []) or []),
            metadata=dict((invoke_config or {}).get("metadata", {}) or {}),
            summary_inputs={
                "problem_id": candidate.get("id", ""),
                "slot": int(candidate.get("_slot", candidate.get("slot", 0)) or 0),
                "repair_type": "code_only",
            },
            output_key="code_only_hotfix",
        )
    except LengthFinishReasonError as exc:
        raise RuntimeError(f"length_limit_error: code_only hotfix hit model token cap: {exc}") from exc
    repaired = dict(candidate)
    repaired["answer"] = str(result.get("answer", repaired.get("answer", "")))
    repaired["solution"] = result.get("solution", repaired.get("solution", ""))
    repaired["code"] = result.get("code", repaired.get("code", ""))
    repaired["code_runtime_mode"] = result.get("code_runtime_mode") or detect_runtime_mode(repaired.get("code", "") or "")
    repaired["evidence_summary"] = result.get("evidence_summary", repaired.get("evidence_summary", ""))
    repaired["repair_rationale"] = result.get("repair_rationale", "")
    repaired["repair_surface"] = "code_only_hotfix"
    _enforce_hotfix_brief(candidate, repaired, synthesis_brief)
    return repaired


def apply_target_only_hotfix(
    candidate: Dict,
    *,
    invariant_bundles: List[Dict] = None,
    synthesis_brief: Dict = None,
    validation_feedback: str = "",
    escalation_signal: str = "",
    invoke_config: Dict = None,
) -> Dict:
    invariant_bundle_block = build_invariant_bundle_block(invariant_bundles or [])
    prompt = build_target_only_hotfix_prompt(
        problem_summary=_problem_summary(candidate),
        invariant_bundle_block=invariant_bundle_block,
        validation_feedback=_combined_hotfix_feedback(validation_feedback, synthesis_brief),
        escalation_signal=escalation_signal,
    )
    context = _build_context(
        slot=int(candidate.get("_slot", candidate.get("slot", 0)) or 0),
        op_type=str(candidate.get("op_type", "mutation") or "mutation"),
        parent_ids=list(candidate.get("parent_ids", []) or []),
        invariant_bundle_block=invariant_bundle_block,
        objective="Repair the target semantics without running a full regenerate path.",
        strategy_summary="Use a narrow target-only hotfix to restore target_quantity_guard.",
        finish_when=[
            "Preserve the exact final target semantics.",
            "Return a repaired statement, answer, solution, and code.",
            "Avoid proxy targets and helper-only objectives.",
        ],
    )
    structured_llm = _repair_llm.with_structured_output(TargetOnlyHotfixSchema, include_raw=True)
    try:
        result = invoke_structured_with_slim_trace(
            structured_llm,
            [
                SystemMessage(content=TARGET_ONLY_HOTFIX_SYSTEM_PROMPT),
                HumanMessage(content=context + "\n\n" + prompt),
            ],
            invoke_config=invoke_config,
            trace_name=((invoke_config or {}).get("run_name") or "deepagent.repair_hotfix.target"),
            tags=list((invoke_config or {}).get("tags", []) or []),
            metadata=dict((invoke_config or {}).get("metadata", {}) or {}),
            summary_inputs={
                "problem_id": candidate.get("id", ""),
                "slot": int(candidate.get("_slot", candidate.get("slot", 0)) or 0),
                "repair_type": "target_only",
            },
            output_key="target_only_hotfix",
        )
    except LengthFinishReasonError as exc:
        raise RuntimeError(f"length_limit_error: target_only hotfix hit model token cap: {exc}") from exc
    repaired = dict(candidate)
    repaired["statement"] = result.get("statement", repaired.get("statement", ""))
    repaired["answer"] = str(result.get("answer", repaired.get("answer", "")))
    repaired["solution"] = result.get("solution", repaired.get("solution", ""))
    repaired["code"] = result.get("code", repaired.get("code", ""))
    repaired["code_runtime_mode"] = result.get("code_runtime_mode") or detect_runtime_mode(repaired.get("code", "") or "")
    repaired["evidence_summary"] = result.get("evidence_summary", repaired.get("evidence_summary", ""))
    repaired["repair_rationale"] = result.get("repair_rationale", "")
    repaired["repair_surface"] = "target_only_hotfix"
    _enforce_hotfix_brief(candidate, repaired, synthesis_brief)
    return repaired


def apply_statement_domain_hotfix(
    candidate: Dict,
    *,
    invariant_bundles: List[Dict] = None,
    synthesis_brief: Dict = None,
    validation_feedback: str = "",
    escalation_signal: str = "",
    invoke_config: Dict = None,
) -> Dict:
    invariant_bundle_block = build_invariant_bundle_block(invariant_bundles or [])
    prompt = build_statement_domain_hotfix_prompt(
        problem_summary=_problem_summary(candidate),
        invariant_bundle_block=invariant_bundle_block,
        validation_feedback=_combined_hotfix_feedback(validation_feedback, synthesis_brief),
        escalation_signal=escalation_signal,
    )
    context = _build_context(
        slot=int(candidate.get("_slot", candidate.get("slot", 0)) or 0),
        op_type=str(candidate.get("op_type", "mutation") or "mutation"),
        parent_ids=list(candidate.get("parent_ids", []) or []),
        invariant_bundle_block=invariant_bundle_block,
        objective="Repair the statement and domain semantics without changing the mathematical family.",
        strategy_summary="Use a narrow statement/domain hotfix instead of full regeneration.",
        finish_when=[
            "Preserve the original domain constraints.",
            "Preserve the original mathematical family and defining system semantics.",
            "Return a repaired statement, answer, solution, and code.",
        ],
    )
    structured_llm = _repair_llm.with_structured_output(StatementDomainHotfixSchema, include_raw=True)
    try:
        result = invoke_structured_with_slim_trace(
            structured_llm,
            [
                SystemMessage(content=STATEMENT_DOMAIN_HOTFIX_SYSTEM_PROMPT),
                HumanMessage(content=context + "\n\n" + prompt),
            ],
            invoke_config=invoke_config,
            trace_name=((invoke_config or {}).get("run_name") or "deepagent.repair_hotfix.statement_domain"),
            tags=list((invoke_config or {}).get("tags", []) or []),
            metadata=dict((invoke_config or {}).get("metadata", {}) or {}),
            summary_inputs={
                "problem_id": candidate.get("id", ""),
                "slot": int(candidate.get("_slot", candidate.get("slot", 0)) or 0),
                "repair_type": "statement_domain",
            },
            output_key="statement_domain_hotfix",
        )
    except LengthFinishReasonError as exc:
        raise RuntimeError(f"length_limit_error: statement_domain hotfix hit model token cap: {exc}") from exc
    repaired = dict(candidate)
    repaired["statement"] = result.get("statement", repaired.get("statement", ""))
    repaired["answer"] = str(result.get("answer", repaired.get("answer", "")))
    repaired["solution"] = result.get("solution", repaired.get("solution", ""))
    repaired["code"] = result.get("code", repaired.get("code", ""))
    repaired["code_runtime_mode"] = result.get("code_runtime_mode") or detect_runtime_mode(repaired.get("code", "") or "")
    repaired["evidence_summary"] = result.get("evidence_summary", repaired.get("evidence_summary", ""))
    repaired["repair_rationale"] = result.get("repair_rationale", "")
    repaired["repair_surface"] = "statement_domain_hotfix"
    _enforce_hotfix_brief(candidate, repaired, synthesis_brief)
    return repaired
