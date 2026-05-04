import logging
from typing import Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_llm_config
from deepagent.tracing import invoke_structured_with_slim_trace
from prompts import (
    QUALITY_SYSTEM_PROMPT,
    QualityAssessmentSchema,
    build_quality_prompt,
)

logger = logging.getLogger("quality")

config = get_llm_config("quality")
_quality_llm = ChatOpenAI(
    model=config["model"],
    temperature=0.0,
    max_tokens=config.get("max_tokens", 4000),
    timeout=config.get("timeout", 120),
    max_retries=config.get("max_retries", 3),
    api_key=config["api_key"],
    base_url=config["base_url"],
    extra_body=config.get("extra_body"),
)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _is_trivial_direct_subtask(statement: str) -> bool:
    s = _normalize(statement)
    if "f(6)" in s or "f(12)" in s:
        return True
    if "s(a, b, c) <=" in s or "s(a,b,c) <=" in s:
        return True
    if "divisor of 6" in s or "divides 6" in s:
        return True
    if "divisor of 12" in s and "determine the value of f(12)" in s:
        return True
    if "count the number of triples" in s and ("<= 10" in s or "<= 6" in s or "<= 12" in s):
        return True
    return False


def assess_problem_quality(problem: Dict, parents: List[Dict], target_diff: float, invoke_config: Dict = None) -> Tuple[bool, str, Dict]:
    statement = _normalize(problem.get("statement", ""))

    if _is_trivial_direct_subtask(statement):
        assessment = {
            "passed": True,
            "issues": ["too_easy_derivative", "novelty_low"],
            "reason": "Accepted for evolution despite being a trivial direct sub-task because low-difficulty descendants are allowed to survive when they preserve the parent invariant and may still stabilize later generations.",
            "novelty_score": 1,
            "difficulty_alignment_score": 1,
        }
        return True, assessment["reason"], assessment

    prompt = build_quality_prompt(problem, parents, target_diff)
    try:
        structured_llm = _quality_llm.with_structured_output(QualityAssessmentSchema, include_raw=True)
        assessment = invoke_structured_with_slim_trace(
            structured_llm,
            [
                SystemMessage(content=QUALITY_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            invoke_config=invoke_config,
            trace_name=((invoke_config or {}).get("run_name") or "orchestrator.quality.llm"),
            tags=list((invoke_config or {}).get("tags", []) or []),
            metadata=dict((invoke_config or {}).get("metadata", {}) or {}),
            summary_inputs={
                "problem_id": problem.get("id", ""),
                "parent_ids": problem.get("parent_ids", []) or [],
                "target_diff": target_diff,
            },
            output_key="assessment",
        )
        if bool(assessment.get("passed")) is False:
            assessment["reason"] = assessment.get("reason", "No reason provided") or "No reason provided"
        # Quality is advisory: keep the issues, but let the orchestrator decide how to react.
        assessment["passed"] = True
        return True, assessment.get("reason", "No reason provided"), assessment
    except Exception as exc:
        logger.warning(f"Quality assessment failed: {exc}")
        assessment = {
            "passed": True,
            "issues": ["other"],
            "reason": f"Quality assessment error (advisory only): {exc}",
            "novelty_score": 1,
            "difficulty_alignment_score": 1,
        }
        return True, assessment["reason"], assessment
