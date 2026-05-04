"""P4 — Grounding & Rescore gate.

Runs AFTER validate_candidates and BEFORE save_generation. Each candidate that
survived the regenerability validator is sent to a single structured LLM call
(GroundingReviewSchema) that performs three tasks at once:

  1. Grounding rewrite of the solution.
  2. Three adversarial probes with per-probe verdicts.
  3. Structural difficulty rescore.

Decisions:
  - accept   -> candidate moves to save_generation with grounded_solution and
                rescored_difficulty applied.
  - rescope  -> statement flagged with rescope_suggestion; caller may attempt a
                minimal rewrite on the next pass. For the MVP we treat rescope
                as a soft failure (routed back through regenerate_failed).
  - reject   -> candidate hard-failed; routed back through regenerate_failed
                with the reviewer's reason as feedback.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_llm_config
from deepagent.tracing import invoke_structured_with_slim_trace
from prompts import (
    GROUNDING_REVIEWER_SYSTEM_PROMPT,
    GroundingReviewSchema,
    build_grounding_review_prompt,
)

logger = logging.getLogger("deep_grounding")

# Mirrors deepagent.graph.runtime._coerce_difficulty_value without importing the graph package here.
_DIFFICULTY_LABEL_TO_SCORE = {
    "easy": 4.0, "medium": 6.0, "hard": 8.0, "superhard": 9.5, "survivor": 0.0,
}


def _coerce_difficulty(value, default: float = 7.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return default
    if text in _DIFFICULTY_LABEL_TO_SCORE:
        return _DIFFICULTY_LABEL_TO_SCORE[text]
    try:
        return float(text)
    except Exception:
        return default

_cfg = get_llm_config("advisor")
_grounding_llm = ChatOpenAI(
    model=_cfg["model"],
    temperature=0.0,
    max_tokens=_cfg.get("max_tokens", 6000),
    timeout=_cfg.get("timeout", 180),
    max_retries=_cfg.get("max_retries", 3),
    api_key=_cfg["api_key"],
    base_url=_cfg["base_url"],
).with_structured_output(GroundingReviewSchema, include_raw=True)


def _render_parent_block(parents: List[Dict]) -> str:
    if not parents:
        return ""
    lines = []
    for idx, parent in enumerate(parents[:2], 1):
        sol = (parent.get("solution") or parent.get("solution_sketch") or "").strip()
        lines.append(
            f"Parent {idx} ({parent.get('id','')}):\n"
            f"  Statement: {(parent.get('statement','') or '').strip()[:1200]}\n"
            f"  Canonical solution: {(sol or '(unavailable)')[:2000]}\n"
            f"  Final answer: {parent.get('answer','')}"
        )
    return "\n\n".join(lines)


def _extract_sandbox_evidence(candidate: Dict, external_evidence: Optional[Dict] = None) -> str:
    """Pull a compact sandbox-evidence blob from candidate fields + optional state evidence.

    `external_evidence` is the per-candidate slice of state['validation_evidence']
    that the orchestrator node injects (the underscore-prefixed
    `_code_execution_assessment` is stripped before save_generation, and the
    state-level validation_evidence is keyed separately, so passing it in
    explicitly avoids the silent-empty-evidence bug C5).
    """
    parts: List[str] = []
    assess = candidate.get("_code_execution_assessment") or candidate.get("code_execution_assessment")
    if assess:
        stdout = str(assess.get("stdout", "") or "").strip()
        if stdout:
            parts.append(f"stdout:\n{stdout[:1600]}")
        exit_code = assess.get("exit_code")
        if exit_code is not None:
            parts.append(f"exit_code: {exit_code}")
    # External evidence takes priority over the candidate's own (possibly empty)
    # validation_evidence dict.
    evidence_payload = external_evidence if external_evidence is not None else candidate.get("validation_evidence")
    if isinstance(evidence_payload, dict) and evidence_payload:
        parts.append("validation_evidence:\n" + json.dumps(evidence_payload, ensure_ascii=False)[:1200])
    summary = str(candidate.get("evidence_summary", "") or "").strip()
    if summary:
        parts.append(f"evidence_summary: {summary[:600]}")
    return "\n\n".join(parts)


def _build_peer_context_block(candidate: Dict[str, Any]) -> str:
    peer_context = dict(candidate.get("orchestrator_peer_context") or {})
    peer_summaries = list(peer_context.get("peer_summaries") or [])
    failed_summaries = list(peer_context.get("failed_slot_summaries") or [])
    if not peer_summaries and not failed_summaries:
        return ""
    lines = [f"Parallel peer context ({peer_context.get('phase', 'unknown_phase')}):"]
    for summary in peer_summaries[:3]:
        lines.append(
            "- slot {slot} | {op_type} | id={problem_id} | axis={axis} | answer={answer} | stmt={stmt}".format(
                slot=summary.get("slot"),
                op_type=summary.get("op_type", ""),
                problem_id=summary.get("problem_id", ""),
                axis=str(summary.get("variation_axis", "") or "")[:60],
                answer=str(summary.get("answer_preview", "") or "")[:60],
                stmt=str(summary.get("statement_preview", "") or "")[:120],
            )
        )
    for summary in failed_summaries[:3]:
        lines.append(
            "- failed slot {slot} | stage={stage} | type={failure_type} | reason={reason}".format(
                slot=summary.get("slot"),
                stage=summary.get("failure_stage", ""),
                failure_type=summary.get("failure_type", ""),
                reason=str(summary.get("failure_reason", "") or "")[:120],
            )
        )
    return "\n".join(lines)


def _deterministic_difficulty_signal(candidate: Dict) -> Dict[str, Any]:
    """Cheap structural signal used as a sanity-check on the LLM rescore (Codex M2).

    Independent of any LLM. Currently combines: solution length, code length,
    presence of symbolic libraries, and existing quality_assessment.score (when
    populated by the validator pipeline). Returns a dict that grounding.py
    surfaces as `deterministic_difficulty_hint` in the result for downstream
    auditing — it is NOT used to override the LLM rescore directly, but a large
    discrepancy is logged so future fitness work can compare both signals.
    """
    sol_len = len((candidate.get("solution") or "").strip())
    code = (candidate.get("code") or "").strip()
    code_len = len(code)
    has_sympy = "sympy" in code or "from sympy" in code
    has_scipy = "scipy" in code or "numpy" in code
    quality = candidate.get("quality_assessment") or {}
    qscore = float(quality.get("score", 0.0) or 0.0)

    # Heuristic: longer solution + symbolic verification → higher.
    score = 5.0
    if sol_len > 1500:
        score += 1.5
    elif sol_len > 600:
        score += 0.7
    if has_sympy:
        score += 1.0
    elif has_scipy:
        score += 0.3
    if code_len > 1200:
        score += 0.5
    if qscore:
        # quality.py scores in [0,1]; map to ±1.0 around the heuristic.
        score += (qscore - 0.5) * 2.0
    score = max(1.0, min(10.0, score))
    return {
        "score": round(score, 2),
        "components": {
            "solution_length": sol_len,
            "code_length": code_len,
            "symbolic_runtime": has_sympy,
            "scientific_runtime": has_scipy,
            "quality_score": qscore,
        },
    }


def _derive_guards(candidate: Dict, synthesis_brief: Optional[Dict]) -> Dict[str, str]:
    brief = synthesis_brief or {}
    return {
        "target_quantity_guard": str(brief.get("target_quantity_guard", "") or ""),
        "relation_guard": str(brief.get("relation_guard", "") or ""),
    }


def ground_and_rescore(
    candidate: Dict,
    parents: List[Dict],
    *,
    synthesis_brief: Optional[Dict] = None,
    invoke_config: Optional[Dict] = None,
    sandbox_evidence: Optional[str] = None,
    op_type: str = "",
    archive_neighbors: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Run the grounding gate on a single candidate.

    Returns a dict with keys:
      - decision : "accept" | "rescope" | "reject"
      - grounded_solution : str (overrides candidate['solution'] on accept)
      - rescored_difficulty : float (LLM rescore, schema-clamped to [1,10])
      - deterministic_difficulty_hint : Dict (independent structural signal)
      - probes : List[{question, verdict}]  (exactly 3 by schema)
      - rescope_suggestion : str
      - reason : str
      - raw : full schema dump for artifact logging
      - llm_failed : bool

    On LLM failure: returns decision='reject' with reason describing the failure.
    Previous "silent accept-as-is" passthrough was removed (Phase 3 QA C3) — a
    failed gate must NOT silently approve the candidate.
    """
    stmt = (candidate.get("statement", "") or "").strip()
    answer = str(candidate.get("answer", "") or "").strip()
    current_solution = (candidate.get("solution") or candidate.get("solution_sketch") or "").strip()
    code = (candidate.get("code", "") or "").strip()
    guards = _derive_guards(candidate, synthesis_brief)
    parent_block = _render_parent_block(parents or [])
    # Caller-supplied sandbox_evidence (from the orchestrator's state-level
    # validation_evidence) takes priority. Falls back to candidate-local fields
    # so unit-test direct calls still work.
    if sandbox_evidence is None:
        sandbox_evidence = _extract_sandbox_evidence(candidate)
    deterministic_hint = _deterministic_difficulty_signal(candidate)
    resolved_op_type = op_type or str(candidate.get("op_type") or candidate.get("type") or "")

    prompt = build_grounding_review_prompt(
        statement=stmt,
        current_solution=current_solution,
        answer=answer,
        verification_code=code,
        sandbox_evidence=sandbox_evidence,
        parent_blocks=parent_block,
        target_quantity_guard=guards["target_quantity_guard"],
        relation_guard=guards["relation_guard"],
        current_difficulty=_coerce_difficulty(candidate.get("difficulty"), default=0.0),
        op_type=resolved_op_type,
        archive_neighbors=archive_neighbors or [],
    )
    peer_block = _build_peer_context_block(candidate)
    if peer_block:
        prompt += "\n\n" + peer_block

    run_name = (invoke_config or {}).get("run_name") or "deepagent.ground_and_rescore"
    try:
        review = invoke_structured_with_slim_trace(
            _grounding_llm,
            [
                SystemMessage(content=GROUNDING_REVIEWER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            invoke_config={**(invoke_config or {}), "run_name": run_name},
            trace_name=f"{run_name}.llm",
            tags=list((invoke_config or {}).get("tags", []) or []) + ["deepagent", "grounding"],
            metadata=dict((invoke_config or {}).get("metadata", {}) or {}),
            summary_inputs={
                "problem_id": candidate.get("id", ""),
                "slot": int(candidate.get("_slot", 0) or 0),
                "op_type": resolved_op_type,
            },
            output_key="review",
        )
    except Exception as exc:
        logger.warning(f"ground_and_rescore LLM failed; rejecting candidate to avoid silent passthrough: {exc}")
        return {
            "decision": "reject",
            "grounded_solution": "",
            "rescored_difficulty": _coerce_difficulty(candidate.get("difficulty"), default=0.0),
            "deterministic_difficulty_hint": deterministic_hint,
            "probes": [],
            "rescope_suggestion": "",
            "novelty_verdict": "novel",
            "novelty_matched_card_id": "",
            "reason": f"grounding LLM failure — gate rejected candidate (no silent accept): {exc}",
            "raw": {},
            "llm_failed": True,
        }

    decision = str(review.get("decision", "reject") or "reject").lower()
    if decision not in {"accept", "rescope", "reject"}:
        decision = "reject"
    grounded = str(review.get("grounded_solution", "") or "").strip()
    # Schema enforces non-empty grounded_solution when accept; double-defense:
    # if somehow empty, demote to reject rather than silently re-using the
    # original ungrounded solution.
    if decision == "accept" and not grounded:
        decision = "reject"
        review = {**review, "decision": "reject", "reason": (review.get("reason", "") or "") + " | [contract] empty grounded_solution after schema validation"}
    # Sprint 3: novelty_verdict='near_duplicate' overrides decision to reject.
    novelty_verdict = str(review.get("novelty_verdict", "novel") or "novel").lower()
    if novelty_verdict not in {"novel", "structural_overlap", "near_duplicate"}:
        novelty_verdict = "novel"
    novelty_matched = str(review.get("novelty_matched_card_id", "") or "").strip()
    if novelty_verdict == "near_duplicate" and decision == "accept":
        decision = "reject"
        review = {
            **review, "decision": "reject",
            "reason": (review.get("reason", "") or "")
                + f" | [novelty] near_duplicate of archive card {novelty_matched or '?'}",
        }
    rescored = review.get("rescored_difficulty", candidate.get("difficulty", 0.0))
    try:
        rescored = float(rescored)
    except Exception:
        rescored = _coerce_difficulty(candidate.get("difficulty"), default=0.0)
    # Schema already validates [1,10]; this is a final defense against type coercion artifacts.
    rescored = max(1.0, min(10.0, rescored))
    # Surface significant LLM↔deterministic discrepancy for downstream auditing.
    discrepancy = abs(rescored - deterministic_hint["score"])
    if discrepancy >= 2.5:
        logger.info(
            f"ground_and_rescore: LLM rescore {rescored:.1f} vs deterministic hint {deterministic_hint['score']:.1f} "
            f"(discrepancy {discrepancy:.1f}) for problem {candidate.get('id','')}"
        )
    return {
        "decision": decision,
        "grounded_solution": grounded,
        "rescored_difficulty": rescored,
        "deterministic_difficulty_hint": deterministic_hint,
        "probes": list(review.get("probes", []) or []),
        "rescope_suggestion": str(review.get("rescope_suggestion", "") or ""),
        "novelty_verdict": novelty_verdict,
        "novelty_matched_card_id": novelty_matched,
        "reason": str(review.get("reason", "") or ""),
        "raw": review,
        "llm_failed": False,
    }
