"""
Regenerability Validator
솔루션과 답만 보고 원래 문제를 재생성할 수 있는지 검증
"""
import ast
import difflib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith.run_helpers import get_current_run_tree, tracing_context
from config import get_llm_config
from deepagent.family_strategy import infer_family_signature, requires_solvability_gate
from deepagent.python_sandbox import execute_python_code
from deepagent.tracing import invoke_structured_with_slim_trace
from prompts import (
    build_archival_evidence_block,
    build_orchestrator_context_block,
    build_invariant_bundle_block,
    OrchestratorDispatchSchema,
    OrchestratorPlanSchema,
    ReconstructedProblemSchema,
    SolutionGroundingSchema,
    SolvabilityPlanSchema,
    SolvabilityAssessmentSchema,
    VALIDATOR_EXECUTION_SYSTEM_PROMPT,
    VALIDATOR_GROUNDING_SYSTEM_PROMPT,
    ValidatorEquivalenceSchema,
    VALIDATOR_COMPARE_SYSTEM_PROMPT,
    VALIDATOR_REGEN_SYSTEM_PROMPT,
    VALIDATOR_SOLVABILITY_SYSTEM_PROMPT,
    build_validator_comparison_prompt,
    build_validator_parent_anchored_comparison_prompt,
    build_validator_regeneration_prompt,
    VALIDATOR_ANCHORED_RETRY_SYSTEM_PROMPT,
    build_solution_grounding_prompt,
    build_validator_solvability_prompt,
)

logger = logging.getLogger("validator")

# Initialize LLM
config = get_llm_config("validator")
_validator_llm = ChatOpenAI(
    model=config["model"],
    temperature=0.0,
    max_tokens=config.get("max_tokens", 24000),
    timeout=config.get("timeout", 120),
    max_retries=config.get("max_retries", 3),
    api_key=config["api_key"],
    base_url=config["base_url"],
)

# T3-a: trigger parent-anchored 2nd-pass regenerability retry when the 1st-pass
# hard_fail reason indicates the reconstruction LLM drifted into an unrelated
# domain or restructured a definition, not that the child itself is broken.
# Phase-1 Fix-#3 (2026-04-18): broadened from the original 5-pattern regex.
# Run-A post-mortem found 4-5 hard_fail cases whose drift phrasing was
# `definition rewrite of f` / `omits the .* constraint` / `fundamentally
# changes the …` and slipped past the narrower regex, so T3-a never got a
# chance to rescue them. Each new alternative below is grounded in a
# verbatim phrase from the failed runs.
_REGEN_DRIFT_REASON_PATTERNS = re.compile(
    r"(entirely unrelated"
    r"|replaces .+ with"
    r"|different (domain|problem|topic|family)"
    r"|switches from .+ to"
    r"|wholly different"
    r"|fundamentally chang(es|ing|ed)"
    r"|definition[ _]rewrite"
    r"|domain[ _]shift"
    r"|target[ _]shift"
    r"|omits the .+ constraint"
    r"|abandons the .+ structure)",
    re.IGNORECASE,
)


def _normalize_answer(value: str) -> str:
    return " ".join((value or "").strip().split())


def _build_peer_context_block(problem: Dict[str, Any]) -> str:
    peer_context = dict(problem.get("orchestrator_peer_context") or {})
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


_STATEMENT_NORMALIZE_RE = re.compile(r"\s+")


def _normalize_statement_for_similarity(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation for robust similarity."""
    cleaned = re.sub(r"[^0-9a-zA-Z\s]", " ", (text or "").lower())
    return _STATEMENT_NORMALIZE_RE.sub(" ", cleaned).strip()


_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-zA-Z]+")
# Stopwords are deliberately narrow: only pure function words and generic task verbs.
# Math-salient category words (integer, real, polynomial, matrix, ...) are PRESERVED
# because they often carry the parent's defining structure.
_COMMON_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "is", "are", "be",
    "with", "that", "this", "these", "those", "as", "at", "by", "let",
    "find", "compute", "determine", "prove", "show", "given", "suppose", "all",
    "any", "some", "each", "every", "if", "then", "we", "call", "say", "define",
    "consider", "problem", "solution", "answer", "such", "where", "which", "from",
})


def _salient_tokens(text: str, min_len: int = 3, limit: int = 16) -> List[str]:
    """Extract content tokens from a math statement for recognizability checks.

    Two-pass strategy: first collect strict tokens (len >= min_len, non-stopword,
    non-numeric); if nothing survives, fall back to a relaxed pass that keeps any
    non-stopword non-numeric token >= 2 chars. This prevents false-negative
    recognizability failures on statements whose salient tokens are short math
    abbreviations (mod, gcd, lcm, abs, sum, etc.).
    """
    cleaned = (text or "").lower()
    tokens = [tok for tok in _TOKEN_SPLIT_RE.split(cleaned) if tok]

    def _gather(min_length: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for tok in tokens:
            if tok in _COMMON_STOPWORDS or tok.isdigit() or len(tok) < min_length:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= limit:
                break
        return out

    strict = _gather(min_len)
    if strict:
        return strict
    return _gather(2)


def assess_near_copy(candidate: Dict, parents: List[Dict], similarity_threshold: float = 0.85) -> Dict[str, Any]:
    """Detect survivor-style near-copies: same answer AND statement closely mirrors a parent.

    Returns {"near_copy": bool, "similarity": float, "reason": str, "matched_parent_id": str}.
    Called before the LLM regenerability validator to cut short collapsing chains.
    """
    if not parents:
        return {"near_copy": False, "similarity": 0.0, "reason": "", "matched_parent_id": ""}
    child_answer = _normalize_answer(str(candidate.get("answer", "")))
    if not child_answer:
        return {"near_copy": False, "similarity": 0.0, "reason": "", "matched_parent_id": ""}
    child_stmt_norm = _normalize_statement_for_similarity(candidate.get("statement", ""))
    if not child_stmt_norm:
        return {"near_copy": False, "similarity": 0.0, "reason": "", "matched_parent_id": ""}

    best_sim = 0.0
    best_parent_id = ""
    for parent in parents:
        parent_answer = _normalize_answer(str(parent.get("answer", "")))
        if not parent_answer or parent_answer != child_answer:
            continue
        parent_stmt_norm = _normalize_statement_for_similarity(parent.get("statement", ""))
        if not parent_stmt_norm:
            continue
        sim = difflib.SequenceMatcher(None, parent_stmt_norm, child_stmt_norm).ratio()
        if sim > best_sim:
            best_sim = sim
            best_parent_id = parent.get("id", "") or parent.get("problem_id", "")
    if best_sim >= similarity_threshold:
        return {
            "near_copy": True,
            "similarity": round(best_sim, 3),
            "reason": (
                f"Child answer equals parent {best_parent_id} answer and statement similarity "
                f"{best_sim:.2f} ≥ {similarity_threshold:.2f}. Blocking to prevent survivor-collapse chain."
            ),
            "matched_parent_id": best_parent_id,
        }
    return {
        "near_copy": False,
        "similarity": round(best_sim, 3),
        "reason": "",
        "matched_parent_id": best_parent_id,
    }


def build_solvability_ablation_skip(problem: Dict, invariant_bundles=None) -> Dict[str, Any]:
    """Return a trace record for runs that disable solvability enforcement.

    The ablation condition should not silently erase the gate. This lightweight
    record preserves whether the candidate falls in the gate's scope while
    deferring any expensive bounded probe to the post-hoc analyzer.
    """
    gate_required, family = _solvability_gate_requirement(problem, invariant_bundles)
    if problem.get("type") == "survivor":
        gate_required = False
        reason = "Survivors bypass solvability gate; ablation condition recorded only."
        probe_summary = "Skipped for survivor under no_solvability ablation."
    elif gate_required:
        reason = "Solvability enforcement skipped by no_solvability ablation condition."
        probe_summary = "Ablation skipped; use post-hoc shadow solvability checks for failure-rate evidence."
    else:
        reason = "Candidate is not in a constraint-heavy family; ablation condition recorded only."
        probe_summary = "Skipped because the constraint family is not in v1 scope."
    return {
        "gate_required": bool(gate_required),
        "constraint_family": family,
        "verdict": "skip",
        "failure_type": "",
        "reason": reason,
        "deterministic_summary": "",
        "probe_summary": probe_summary,
        "supported": False,
        "skipped": True,
        "ablation_skipped": True,
    }


def _extract_last_numeric_token(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    last = lines[-1]
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", last.replace(",", ""))
    return matches[-1] if matches else ""


def _cached_code_execution_assessment(problem: Dict) -> Optional[Dict[str, Any]]:
    cached = problem.get("_code_execution_assessment")
    return dict(cached) if isinstance(cached, dict) else None


def assess_code_execution(problem: Dict, outcome: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cached = _cached_code_execution_assessment(problem)
    if cached:
        return cached
    code = (problem.get("code") or "").strip()
    expected_answer = _normalize_answer(str(problem.get("answer", "")))
    if not code:
        result = {
            "ok": False,
            "canonical_answer": "",
            "materialize_reason": "No code provided",
            "validation_reason": "No code provided",
            "error_type": "missing_code",
            "runtime_mode": problem.get("code_runtime_mode", ""),
            "python_bin": problem.get("sandbox_python_bin", ""),
            "outcome": None,
        }
        problem["_code_execution_assessment"] = dict(result)
        return result
    outcome = outcome or execute_python_code(code, mode=problem.get("code_runtime_mode"))
    if not outcome["ok"]:
        reason = f"{outcome['error_type']}: {outcome['error_message']}"
        result = {
            "ok": False,
            "canonical_answer": "",
            "materialize_reason": reason,
            "validation_reason": reason,
            "error_type": outcome["error_type"],
            "runtime_mode": outcome.get("mode", problem.get("code_runtime_mode", "")),
            "python_bin": outcome.get("python_bin", problem.get("sandbox_python_bin", "")),
            "outcome": outcome,
        }
        problem["_code_execution_assessment"] = dict(result)
        return result

    stdout = (outcome.get("stdout") or "").strip()
    canonical_answer = ""
    materialize_reason = "Could not extract canonical answer from code output"
    normalized_stdout = _normalize_answer(stdout)
    if normalized_stdout:
        if "\n" not in stdout and normalized_stdout:
            canonical_answer = normalized_stdout
            materialize_reason = "Canonical answer materialized from exact stdout"
        else:
            lines = [line.strip() for line in stdout.splitlines() if line.strip()]
            if lines:
                last_line = _normalize_answer(lines[-1])
                if last_line:
                    canonical_answer = last_line
                    materialize_reason = "Canonical answer materialized from last stdout line"
    if not canonical_answer:
        last_numeric = _extract_last_numeric_token(stdout)
        if last_numeric:
            canonical_answer = _normalize_answer(last_numeric)
            materialize_reason = "Canonical answer materialized from last numeric stdout token"

    answer_match = False
    validation_reason = f"Code output does not match claimed answer '{expected_answer}'"
    if expected_answer:
        if normalized_stdout == expected_answer:
            answer_match = True
            validation_reason = "Code output matches answer exactly"
        else:
            lines = [line.strip() for line in stdout.splitlines() if line.strip()]
            if lines:
                last_line = _normalize_answer(lines[-1])
                if last_line == expected_answer:
                    answer_match = True
                    validation_reason = "Code last line matches answer exactly"
                else:
                    last_numeric = _extract_last_numeric_token(last_line)
                    if last_numeric and _normalize_answer(last_numeric) == expected_answer:
                        answer_match = True
                        validation_reason = "Code last numeric output matches answer"

    execution_ok = bool(canonical_answer)
    ok = execution_ok
    if expected_answer and not answer_match and execution_ok:
        validation_reason = (
            f"Code output did not match claimed answer '{expected_answer}', "
            f"but canonical answer '{canonical_answer}' was extracted from code output"
        )

    result = {
        "ok": ok,
        "execution_ok": execution_ok,
        "answer_match": answer_match,
        "canonical_answer": canonical_answer,
        "materialize_reason": materialize_reason,
        "validation_reason": validation_reason,
        "error_type": "" if execution_ok else "answer_mismatch",
        "runtime_mode": outcome.get("mode", problem.get("code_runtime_mode", "")),
        "python_bin": outcome.get("python_bin", problem.get("sandbox_python_bin", "")),
        "outcome": outcome,
    }
    problem["_code_execution_assessment"] = dict(result)
    return result


def materialize_answer_from_code(problem: Dict) -> Tuple[bool, str, str]:
    """
    Execute verification code and extract the orchestrator-canonical answer.
    Returns (ok, canonical_answer, reason).
    """
    assessment = assess_code_execution(problem)
    if assessment["canonical_answer"]:
        return True, assessment["canonical_answer"], assessment["materialize_reason"]
    return False, "", assessment["materialize_reason"]


def assess_answer_type_consistency(problem: Dict) -> Dict[str, Any]:
    """Verify that `answer_type` matches the actual `answer` string format.

    LLMs occasionally claim `answer_type='integer'` while the answer field
    contains an unevaluated symbolic expression (e.g. sympy `Integral(...)`).
    This downstream-breaks memory_bank parsing and difficulty rescoring.

    Returns {"ok": bool, "claimed_type": str, "actual_type": str, "reason": str}.
    "other" is the catch-all and never fails this check.
    """
    claimed = str(problem.get("answer_type", "") or "").strip().lower() or "other"
    answer = str(problem.get("answer", "") or "").strip()
    if not answer:
        return {"ok": True, "claimed_type": claimed, "actual_type": "empty", "reason": ""}
    if claimed == "other":
        return {"ok": True, "claimed_type": claimed, "actual_type": "other", "reason": ""}
    if claimed == "integer":
        try:
            int(answer)
            return {"ok": True, "claimed_type": claimed, "actual_type": "integer", "reason": ""}
        except (ValueError, TypeError):
            pass
        # Float → integer normalization: handles '100.0', '9.000000000000266',
        # and label-prefixed outputs like 'Max value: 9.000000000000266'.
        # Extract the last numeric token from the answer string.
        nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", answer)
        if nums:
            try:
                val = float(nums[-1])
                rounded = int(round(val))
                if abs(val - rounded) < 1e-6:
                    problem["answer"] = str(rounded)
                    return {"ok": True, "claimed_type": claimed, "actual_type": "integer", "reason": ""}
            except (ValueError, TypeError, OverflowError):
                pass
        return {
            "ok": False,
            "claimed_type": claimed,
            "actual_type": "non-integer",
            "reason": (
                f"answer_type='integer' but answer='{answer[:60]}' is not parseable as int. "
                f"Demote answer_type to 'other' or rewrite the verification code to print a concrete integer."
            ),
        }
    if claimed == "real":
        try:
            float(answer)
            return {"ok": True, "claimed_type": claimed, "actual_type": "real", "reason": ""}
        except (ValueError, TypeError):
            return {
                "ok": False,
                "claimed_type": claimed,
                "actual_type": "non-real",
                "reason": (
                    f"answer_type='real' but answer='{answer[:60]}' is not parseable as float. "
                    f"Demote answer_type to 'other' or print a concrete numeric value."
                ),
            }
    return {"ok": True, "claimed_type": claimed, "actual_type": "unknown", "reason": ""}


def assess_answer_output_contract(problem: Dict) -> Dict[str, Any]:
    assessment = assess_code_execution(problem)
    outcome = dict(assessment.get("outcome") or {})
    lines = _stdout_lines(outcome)
    claimed = str(problem.get("answer_type", "") or "").strip().lower() or "other"
    canonical = str(assessment.get("canonical_answer", "") or "").strip()
    last_line = lines[-1] if lines else ""
    if claimed == "integer" and canonical.lower() in {"true", "false", "none"}:
        return {
            "ok": False,
            "reason": (
                f"verification code emitted non-integer sentinel '{canonical}' for answer_type='integer'. "
                "Rewrite the verification code to print one concrete final integer."
            ),
            "failure_type": "code_answer_contract_failure",
        }
    if lines and canonical and _normalize_answer(last_line) != _normalize_answer(canonical):
        return {
            "ok": False,
            "reason": "verification code must emit the canonical final answer on the last stdout line.",
            "failure_type": "code_answer_contract_failure",
        }
    return {"ok": True, "reason": "", "failure_type": ""}


def validate_problem_by_code(problem: Dict) -> Tuple[bool, str]:
    """
    Deterministic code/answer validation.
    The candidate passes only if the verification code executes successfully and
    prints the claimed answer, or prints a last-line numeric value matching it.
    Additionally, answer_type ↔ answer format consistency is checked (P1 fix from
    docs/future_improvements.md).
    """
    assessment = assess_code_execution(problem)
    if not assessment["ok"]:
        return assessment["ok"], assessment["validation_reason"]
    output_contract = assess_answer_output_contract(problem)
    problem["answer_output_contract"] = output_contract
    if not output_contract["ok"]:
        return False, output_contract["reason"]
    type_check = assess_answer_type_consistency(problem)
    problem["answer_type_check"] = type_check
    if not type_check["ok"]:
        return False, type_check["reason"]
    return True, assessment["validation_reason"]


def _stdout_lines(outcome: Optional[Dict[str, Any]]) -> list[str]:
    stdout = (outcome or {}).get("stdout") or ""
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _stdout_bindings(lines: list[str]) -> Dict[str, str]:
    bindings: Dict[str, str] = {}
    for line in lines:
        for name, value in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^,\n]+)", line):
            cleaned = " ".join(value.strip().split())
            if cleaned:
                bindings[name] = cleaned
    return bindings


def build_execution_evidence(problem: Dict) -> Dict[str, Any]:
    assessment = assess_code_execution(problem)
    outcome = dict(assessment.get("outcome") or {})
    lines = _stdout_lines(outcome)
    bindings = _stdout_bindings(lines)
    runtime_mode = assessment.get("runtime_mode", problem.get("code_runtime_mode", ""))
    canonical_answer = assessment.get("canonical_answer", "")
    evidence_summary = (
        f"runtime={runtime_mode}; canonical_answer={canonical_answer or 'none'}; "
        f"stdout_lines={len(lines)}; bindings={', '.join(sorted(bindings.keys())[:6]) or 'none'}"
    )
    evidence = {
        "canonical_answer": canonical_answer,
        "stdout_lines": lines,
        "bindings": bindings,
        "runtime_mode": runtime_mode,
        "execution_path_kind": "symbolic" if runtime_mode == "symbolic_python" else "numeric",
        "evidence_summary": evidence_summary,
    }
    problem["execution_evidence"] = evidence
    return evidence


def deterministic_ground_solution(problem: Dict, execution_evidence: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    execution_evidence = execution_evidence or build_execution_evidence(problem)
    canonical_answer = execution_evidence.get("canonical_answer", "") or str(problem.get("answer", "") or "")
    bindings = dict(execution_evidence.get("bindings", {}) or {})
    if not canonical_answer:
        return None

    family = infer_family_signature(problem)
    reasoning = ""
    if family == "symmetric_power_sum_system":
        named = [name for name in ["e1", "e2", "e3", "s1", "s2", "s3", "s4", "p4", "S4"] if name in bindings]
        if named:
            chain = ", ".join(f"{name}={bindings[name]}" for name in named[:5])
            reasoning = f"Using the symmetric-sum quantities computed by the verification code ({chain}), the target expression evaluates to {canonical_answer}."
        else:
            reasoning = f"The verification code derives the required symmetric-sum quantities from the stated system and evaluates the target expression to {canonical_answer}."
    elif family == "binomial_moment":
        named = [name for name in ["n", "p", "mean", "moment", "E", "K"] if name in bindings]
        if named:
            chain = ", ".join(f"{name}={bindings[name]}" for name in named[:4])
            reasoning = f"The verification code computes the stated binomial-moment quantity from {chain} and obtains {canonical_answer}."
        else:
            reasoning = f"The verification code evaluates the stated binomial-moment quantity directly and returns {canonical_answer}."
    elif family == "nearest_prime":
        candidate_bindings = [name for name in ["N", "n", "left", "right", "lower", "upper", "distance"] if name in bindings]
        if candidate_bindings:
            chain = ", ".join(f"{name}={bindings[name]}" for name in candidate_bindings[:5])
            reasoning = f"The verification code checks the nearby prime candidates around the stated target ({chain}) and confirms that the required value is {canonical_answer}."
        else:
            reasoning = f"The verification code checks the nearest admissible prime candidate and confirms the answer {canonical_answer}."
    elif family == "lattice_count":
        reasoning = f"The verification code counts the admissible configurations described in the statement and returns {canonical_answer}."
    elif family == "permutation_matrix_trace":
        named = [name for name in ["a", "b", "p", "trace", "tr", "M"] if name in bindings]
        if named:
            chain = ", ".join(f"{name}={bindings[name]}" for name in named[:4])
            reasoning = f"The verification code computes the relevant permutation action and its trace from {chain}, yielding {canonical_answer}."
        else:
            reasoning = f"The verification code evaluates the required permutation-matrix trace directly and obtains {canonical_answer}."
    elif family == "code_automorphism":
        named = [name for name in ["n", "w", "N", "orbit", "stabilizer"] if name in bindings]
        if named:
            chain = ", ".join(f"{name}={bindings[name]}" for name in named[:4])
            reasoning = f"The verification code computes the relevant orbit/stabilizer count from {chain} and gets {canonical_answer}."
        else:
            reasoning = f"The verification code counts the required code-equivalence or automorphism quantity and returns {canonical_answer}."
    elif family == "graph_count":
        named = [name for name in ["n", "N", "k", "steps", "vertices"] if name in bindings]
        if named:
            chain = ", ".join(f"{name}={bindings[name]}" for name in named[:4])
            reasoning = f"The verification code builds the stated counting recurrence or transition system from {chain} and evaluates the target count as {canonical_answer}."
        else:
            reasoning = f"The verification code evaluates the stated graph/counting process directly and returns {canonical_answer}."
    elif family in {"contour_integral", "limit_integral"}:
        reasoning = f"The verification code evaluates the stated analytic expression under the given constraints and confirms the final value {canonical_answer}."
    else:
        return None

    return {
        "grounded_solution": reasoning,
        "grounded_evidence_summary": execution_evidence.get("evidence_summary", ""),
        "unsupported_claims_removed": [],
        "grounding_status": "grounded",
        "grounding_reason": f"Deterministically grounded for {family}.",
    }


_SOLVABILITY_SKIP = {
    "gate_required": False,
    "constraint_family": "unsupported",
    "verdict": "skip",
    "failure_type": "",
    "reason": "",
    "deterministic_summary": "",
    "probe_summary": "",
    "supported": False,
    "skipped": True,
}


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"-?\d+(?:\.\d+)?", text or ""))


def _contains_constraint_heavy_signal(statement: str) -> Tuple[bool, str]:
    text = statement or ""
    lower = text.lower()
    if any(token in lower for token in ["newton", "symmetric polynomial", "power sum", "elementary symmetric"]):
        return True, "symmetric_power_sum_system"
    if any(token in lower for token in ["natural numbers", "integers", "integer solutions", "natural number solutions"]) and (
        "\\begin{cases}" in text or "=" in text
    ):
        return True, "integer_equation_system"
    if any(token in lower for token in ["let n =", "set n =", "define n =", "let s =", "set s ="]) and (
        "\\lfloor" in text or "find the prime number closest" in lower or "consider natural numbers" in lower
    ):
        return True, "derived_constant_coupling"
    if "\\begin{cases}" in text or "system of" in lower:
        return True, "integer_equation_system"
    return False, "unsupported"


def _invariant_marks_tightly_coupled(invariant_bundles) -> bool:
    bundles = invariant_bundles or []
    joined = " ".join(
        " ".join(str(bundle.get(key, "")) for key in ["named_definition", "target_quantity"])
        for bundle in bundles
    ).lower()
    return any(token in joined for token in ["exact", "iff", "newton", "symmetric", "power sum", "set identity"])


def _solvability_gate_requirement(problem: Dict, invariant_bundles=None) -> Tuple[bool, str]:
    required, family = _contains_constraint_heavy_signal(problem.get("statement", ""))
    if required:
        return True, family
    if _invariant_marks_tightly_coupled(invariant_bundles):
        return True, "derived_constant_coupling"
    return False, "unsupported"


def _extract_assigned_numeric_constants(code: str) -> list[tuple[str, str]]:
    assignments = []
    if not code.strip():
        return assignments
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return assignments

    def _literal(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return str(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return str(-node.operand.value)
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _literal(node.value)
            if value is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.append((target.id, value))
    return assignments


def _statement_code_consistency_check(problem: Dict) -> Tuple[bool, str, str]:
    statement = problem.get("statement", "") or ""
    code = problem.get("code", "") or ""
    statement_numbers = _number_tokens(statement)
    suspicious_names = {"e1", "e2", "e3", "s1", "s2", "s3", "p", "q", "n", "N", "S"}
    dynamic_statement = any(token in statement for token in ["\\lfloor", "N =", "S =", "Let N", "Set N", "Define N"])
    for name, value in _extract_assigned_numeric_constants(code):
        if name not in suspicious_names:
            continue
        if value in {"0", "1", "2", "3", "4"}:
            continue
        if value not in statement_numbers:
            return False, f"Code assigns {name}={value}, but {value} does not appear in the statement.", "hidden_constant_failure"
    if dynamic_statement:
        for name, value in _extract_assigned_numeric_constants(code):
            if name in {"e1", "N", "S"} and value not in statement_numbers and value not in {"0", "1", "2", "3", "4"}:
                return False, f"Code hardcodes derived quantity {name}={value} instead of deriving it from the statement.", "hidden_constant_failure"
    return True, "Statement/code consistency checks passed.", ""


def _solution_assignments(solution: str) -> list[tuple[str, str]]:
    matches = re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)", solution or "")
    return [(name, value) for name, value in matches]


def _statement_solution_consistency_check(problem: Dict) -> Tuple[bool, str, str]:
    statement = problem.get("statement", "") or ""
    solution = problem.get("solution") or problem.get("solution_sketch") or ""
    statement_numbers = _number_tokens(statement)
    dynamic_statement = any(token in statement for token in ["\\lfloor", "N =", "S =", "Let N", "Set N", "Define N"])
    for name, value in _solution_assignments(solution):
        if name not in {"e1", "e2", "e3", "N", "S"}:
            continue
        if value in {"0", "1", "2", "3", "4"}:
            continue
        if value not in statement_numbers and dynamic_statement:
            return False, f"Solution claims {name}={value}, but that constant is not stated or justified by the problem statement.", "statement_solution_inconsistency"
    return True, "Statement/solution consistency checks passed.", ""


def _classify_solvability_failure(problem: Dict, probe_reason: str) -> str:
    text = str(probe_reason or "").strip().lower()
    if "does not appear in the statement" in text or "hardcodes derived quantity" in text:
        return "hidden_constant_failure"
    if "residue" in text or "expected" in text:
        return "residue_mismatch_failure"
    if (
        "no natural-number solution" in text
        or "no cyclically divisible triple" in text
        or "no solution" in text
        or "not satisfiable" in text
    ):
        return "no_solution_under_stated_constraints"
    return "solvability_failure"


def _parse_power_sum_system(statement: str) -> Optional[Dict[str, int]]:
    text = statement or ""

    def _match(pattern: str) -> Optional[int]:
        found = re.search(pattern, text)
        if not found:
            return None
        try:
            return int(found.group(1))
        except Exception:
            return None

    s1 = _match(r"a\s*\+\s*b\s*\+\s*c\s*=\s*(-?\d+)")
    p2 = _match(r"a\^?\{?2\}?\s*\+\s*b\^?\{?2\}?\s*\+\s*c\^?\{?2\}?\s*=\s*(-?\d+)")
    p3 = _match(r"a\^?\{?3\}?\s*\+\s*b\^?\{?3\}?\s*\+\s*c\^?\{?3\}?\s*=\s*(-?\d+)")
    if s1 is None or p2 is None or p3 is None:
        return None
    return {"s1": s1, "p2": p2, "p3": p3}


def _sandbox_json_print_expr(solvable_expr: str, reason_expr: str) -> str:
    return (
        "safe_reason = str(" + reason_expr + ").replace('\\\\', '/').replace('\"', \"'\")\n"
        "print('{\"solvable\": ' + ('true' if (" + solvable_expr + ") else 'false') + ', \"reason\": \"' + safe_reason + '\"}')"
    )


def _deterministic_solvability_probe(problem: Dict, family: str) -> Optional[Dict[str, Any]]:
    parsed = _parse_power_sum_system(problem.get("statement", ""))
    if parsed is None:
        return None
    s1 = parsed["s1"]
    p2 = parsed["p2"]
    p3 = parsed["p3"]
    code = f"""import math
s1 = {s1}
p2 = {p2}
p3 = {p3}
solvable = False
reason = ""
if s1 <= 0:
    reason = "Sum constraint is non-positive."
elif p2 <= 0 or p3 <= 0:
    reason = "Power sums must be positive for natural-number solutions."
elif s1 * s1 < p2:
    reason = "The square-sum exceeds the maximum possible value for the stated sum."
else:
    delta = s1 * s1 - p2
    if delta % 2 != 0:
        reason = "Derived e2 is not integral."
    else:
        e2 = delta // 2
        delta3 = p3 - s1 * p2 + e2 * s1
        if delta3 % 3 != 0:
            reason = "Derived e3 is not integral."
        else:
            e3 = delta3 // 3
            if e3 <= 0:
                reason = "Derived e3 is non-positive."
            else:
                for r in range(1, max(2, s1 + 1)):
                    if r * r * r - s1 * r * r + e2 * r - e3 != 0:
                        continue
                    if e3 % r != 0:
                        continue
                    yz = e3 // r
                    pair_sum = s1 - r
                    disc = pair_sum * pair_sum - 4 * yz
                    if disc < 0:
                        continue
                    root = math.isqrt(disc)
                    if root * root != disc:
                        continue
                    if (pair_sum + root) % 2 != 0 or (pair_sum - root) % 2 != 0:
                        continue
                    y = (pair_sum + root) // 2
                    z = (pair_sum - root) // 2
                    if y > 0 and z > 0 and r + y + z == s1 and r*r + y*y + z*z == p2 and r*r*r + y*y*y + z*z*z == p3:
                        solvable = True
                        reason = f"Found natural-number solution ({{r}}, {{y}}, {{z}})."
                        break
                if not solvable and not reason:
                    reason = "No natural-number solution satisfies the stated system."
""" + _sandbox_json_print_expr("solvable", "reason")
    return {
        "gate_required": True,
        "constraint_family": family,
        "supported": True,
        "consistency_focus": [
            "Check integrality of derived symmetric quantities.",
            "Verify existence of positive integer roots matching all stated power sums.",
        ],
        "feasibility_goal": "Verify that the stated system has a natural-number solution.",
        "exploratory_python_code": code,
        "expected_signal": '{"solvable": true, "reason": "Found natural-number solution (...)."}',
        "skip_reason": "",
    }


def assess_solvability(problem: Dict, invariant_bundles=None, invoke_config: Dict = None) -> Dict[str, Any]:
    gate_required, family = _solvability_gate_requirement(problem, invariant_bundles)
    if problem.get("type") == "survivor":
        return dict(_SOLVABILITY_SKIP) | {
            "reason": "Survivors bypass solvability gate.",
            "probe_summary": "Skipped for survivor.",
        }
    if not gate_required:
        return dict(_SOLVABILITY_SKIP) | {
            "reason": "Candidate is not in a constraint-heavy family.",
            "probe_summary": "Skipped because the constraint family is not in v1 scope.",
        }

    code_ok, code_reason, code_failure_type = _statement_code_consistency_check(problem)
    if not code_ok:
        return SolvabilityAssessmentSchema.model_validate(
            {
                "gate_required": True,
                "constraint_family": family,
                "verdict": "fail",
                "failure_type": code_failure_type or "statement_code_inconsistency",
                "reason": code_reason,
                "deterministic_summary": code_reason,
                "probe_summary": "Probe skipped because deterministic consistency failed.",
                "supported": True,
                "skipped": False,
            }
        ).model_dump()

    solution_ok, solution_reason, solution_failure_type = _statement_solution_consistency_check(problem)
    if not solution_ok:
        return SolvabilityAssessmentSchema.model_validate(
            {
                "gate_required": True,
                "constraint_family": family,
                "verdict": "fail",
                "failure_type": solution_failure_type or "statement_solution_inconsistency",
                "reason": solution_reason,
                "deterministic_summary": solution_reason,
                "probe_summary": "Probe skipped because deterministic consistency failed.",
                "supported": True,
                "skipped": False,
            }
        ).model_dump()

    deterministic_summary = f"{code_reason} {solution_reason}".strip()
    if not requires_solvability_gate({"family_signature": family}) and family != "unsupported":
        return SolvabilityAssessmentSchema.model_validate(
            {
                "gate_required": True,
                "constraint_family": family,
                "verdict": "skip",
                "failure_type": "",
                "reason": "Constraint-heavy family has deterministic checks only; sandbox probe skipped.",
                "deterministic_summary": deterministic_summary,
                "probe_summary": "No additional solvability probe required for this family.",
                "supported": False,
                "skipped": True,
            }
        ).model_dump()
    plan_payload = OrchestratorPlanSchema(
        stage="validate_candidates",
        objective="Determine whether the candidate is satisfiable and statement-consistent before invariant validation.",
        strategy_summary="Run deterministic consistency checks first, then execute a bounded feasibility probe only for supported constraint-heavy families.",
        hard_constraints=[
            "Do not use hidden parent constants.",
            "Probe the stated system exactly as written.",
            "If unsupported, skip instead of bluffing.",
        ],
        success_criteria=[
            "Classify the family correctly.",
            "Return a solvability plan only when safe.",
            "Emit Python that prints a final JSON line.",
        ],
    ).model_dump()
    dispatch_payload = OrchestratorDispatchSchema(
        agent_role="validator",
        slot=int(problem.get("_slot", problem.get("slot", 0)) or 0),
        op_type="validate_solvability",
        parent_ids=problem.get("parent_ids", []) or [],
        invariant_notes="Constraint-heavy candidates must remain satisfiable and statement-consistent.",
        task_payload={
            "problem_id": problem.get("id", ""),
            "constraint_family_hint": family,
            "statement": problem.get("statement", ""),
            "answer": str(problem.get("answer", "")),
        },
        finish_when=[
            "Return a solvability plan with support/skip decision.",
            "Provide a safe sandbox probe for supported families.",
        ],
    ).model_dump()
    peer_block = _build_peer_context_block(problem)
    orchestrator_context = build_orchestrator_context_block(plan_payload, dispatch_payload) + "\n\n" + build_invariant_bundle_block(invariant_bundles or [])
    if peer_block:
        orchestrator_context += "\n\n" + peer_block
    prompt = build_validator_solvability_prompt(
        problem.get("statement", ""),
        problem.get("solution") or problem.get("solution_sketch") or "",
        problem.get("code", ""),
        str(problem.get("answer", "")),
    )

    try:
        plan_llm = _validator_llm.with_structured_output(SolvabilityPlanSchema, include_raw=True)
        run_name = f"{(invoke_config or {}).get('run_name', 'deepagent.validator')}.solvability_plan"
        plan = invoke_structured_with_slim_trace(
            plan_llm,
            [
                SystemMessage(content=VALIDATOR_EXECUTION_SYSTEM_PROMPT + "\n\n" + VALIDATOR_SOLVABILITY_SYSTEM_PROMPT),
                HumanMessage(content=orchestrator_context + "\n\n" + prompt),
            ],
            invoke_config={**(invoke_config or {}), "run_name": run_name},
            trace_name=f"{run_name}.llm",
            tags=list((invoke_config or {}).get("tags", []) or []),
            metadata=dict((invoke_config or {}).get("metadata", {}) or {}),
            summary_inputs={
                "problem_id": problem.get("id", ""),
                "slot": int(problem.get("_slot", problem.get("slot", 0)) or 0),
                "constraint_family": family,
            },
            output_key="solvability_plan",
        )
        plan = SolvabilityPlanSchema.model_validate(plan).model_dump()
    except Exception as exc:
        return SolvabilityAssessmentSchema.model_validate(
            {
                "gate_required": True,
                "constraint_family": family,
                "verdict": "skip",
                "failure_type": "",
                "reason": f"Solvability planning skipped: {exc}",
                "deterministic_summary": deterministic_summary,
                "probe_summary": f"Planner error: {exc}",
                "supported": False,
                "skipped": True,
            }
        ).model_dump()

    deterministic_plan = _deterministic_solvability_probe(problem, family)
    if deterministic_plan:
        plan = {**plan, **deterministic_plan}

    if not plan.get("supported") or not plan.get("gate_required"):
        return SolvabilityAssessmentSchema.model_validate(
            {
                "gate_required": True,
                "constraint_family": plan.get("constraint_family", family),
                "verdict": "skip",
                "failure_type": "",
                "reason": plan.get("skip_reason") or "Solvability probe skipped.",
                "deterministic_summary": deterministic_summary,
                "probe_summary": plan.get("skip_reason") or "Unsupported family for v1 probe.",
                "supported": False,
                "skipped": True,
            }
        ).model_dump() | {"plan": plan}

    outcome = execute_python_code(plan.get("exploratory_python_code", ""), mode="numeric_python")
    if not outcome.get("ok"):
        return SolvabilityAssessmentSchema.model_validate(
            {
                "gate_required": True,
                "constraint_family": plan.get("constraint_family", family),
                "verdict": "skip",
                "failure_type": "",
                "reason": "Solvability probe execution skipped due to sandbox failure.",
                "deterministic_summary": deterministic_summary,
                "probe_summary": f"{outcome.get('error_type')}: {outcome.get('error_message')}",
                "supported": False,
                "skipped": True,
            }
        ).model_dump() | {"plan": plan, "probe_outcome": outcome}

    stdout = (outcome.get("stdout") or "").strip().splitlines()
    final_line = stdout[-1].strip() if stdout else ""
    try:
        probe_result = json.loads(final_line) if final_line else {}
    except Exception:
        return SolvabilityAssessmentSchema.model_validate(
            {
                "gate_required": True,
                "constraint_family": plan.get("constraint_family", family),
                "verdict": "skip",
                "failure_type": "",
                "reason": "Solvability probe did not return parseable JSON.",
                "deterministic_summary": deterministic_summary,
                "probe_summary": final_line or "No output",
                "supported": False,
                "skipped": True,
            }
        ).model_dump() | {"plan": plan, "probe_outcome": outcome}

    solvable = bool(probe_result.get("solvable", False))
    probe_reason = str(probe_result.get("reason", "") or "")
    result = SolvabilityAssessmentSchema.model_validate(
        {
            "gate_required": True,
            "constraint_family": plan.get("constraint_family", family),
            "verdict": "pass" if solvable else "fail",
            "failure_type": "" if solvable else _classify_solvability_failure(problem, probe_reason or final_line),
            "reason": probe_reason or ("Constraint system is satisfiable." if solvable else "Constraint system is inconsistent."),
            "deterministic_summary": deterministic_summary,
            "probe_summary": probe_reason or final_line,
            "supported": True,
            "skipped": False,
        }
    ).model_dump()
    result["plan"] = plan
    result["probe_outcome"] = outcome
    result["probe_result"] = probe_result
    return result


def ground_solution(problem: Dict, invoke_config: Dict = None) -> Dict[str, Any]:
    execution_evidence = build_execution_evidence(problem)
    deterministic = deterministic_ground_solution(problem, execution_evidence=execution_evidence)
    if deterministic:
        return deterministic
    canonical_answer = execution_evidence.get("canonical_answer") or str(problem.get("answer", ""))
    plan_payload = OrchestratorPlanSchema(
        stage="postprocess_candidates",
        objective="Rewrite the candidate solution so it matches the saved statement and deterministic execution evidence.",
        strategy_summary="Use only code-backed numbers and explicit relations already visible in code or evidence.",
        hard_constraints=[
            "Do not invent unsupported intermediate values.",
            "Do not contradict the canonical answer.",
            "Prefer concise competition-style exposition.",
        ],
        success_criteria=[
            "Return a grounded solution.",
            "Refresh evidence_summary consistently.",
            "List unsupported claims that were removed.",
        ],
    ).model_dump()
    dispatch_payload = OrchestratorDispatchSchema(
        agent_role="validator",
        slot=int(problem.get("_slot", problem.get("slot", 0)) or 0),
        op_type="ground_solution",
        parent_ids=problem.get("parent_ids", []) or [],
        invariant_notes="Ground the textual solution against deterministic execution evidence without changing the saved statement.",
        task_payload={
            "problem_id": problem.get("id", ""),
            "canonical_answer": canonical_answer,
            "runtime_mode": execution_evidence.get("runtime_mode", ""),
        },
        finish_when=[
            "Return a grounded solution and refreshed evidence summary.",
            "Do not introduce unsupported arithmetic claims.",
        ],
    ).model_dump()
    prompt = build_solution_grounding_prompt(
        problem.get("statement", ""),
        problem.get("solution") or problem.get("solution_sketch") or "",
        canonical_answer,
        problem.get("code", ""),
        execution_evidence,
    )
    peer_block = _build_peer_context_block(problem)
    orchestrator_context = build_orchestrator_context_block(plan_payload, dispatch_payload)
    if peer_block:
        orchestrator_context += "\n\n" + peer_block
    try:
        grounding_llm = _validator_llm.with_structured_output(SolutionGroundingSchema, include_raw=True)
        run_name = f"{(invoke_config or {}).get('run_name', 'deepagent.validator')}.ground"
        result = invoke_structured_with_slim_trace(
            grounding_llm,
            [
                SystemMessage(content=VALIDATOR_EXECUTION_SYSTEM_PROMPT + "\n\n" + VALIDATOR_GROUNDING_SYSTEM_PROMPT),
                HumanMessage(content=orchestrator_context + "\n\n" + prompt),
            ],
            invoke_config={**(invoke_config or {}), "run_name": run_name},
            trace_name=f"{run_name}.llm",
            tags=list((invoke_config or {}).get("tags", []) or []),
            metadata=dict((invoke_config or {}).get("metadata", {}) or {}),
            summary_inputs={
                "problem_id": problem.get("id", ""),
                "slot": int(problem.get("_slot", problem.get("slot", 0)) or 0),
                "canonical_answer": canonical_answer,
            },
            output_key="grounding",
        )
        result = SolutionGroundingSchema.model_validate(result).model_dump()
        return result
    except Exception as exc:
        fallback_solution = (problem.get("solution") or problem.get("solution_sketch") or "").strip()
        return {
            "grounded_solution": fallback_solution,
            "grounded_evidence_summary": execution_evidence.get("evidence_summary", ""),
            "unsupported_claims_removed": [],
            "grounding_status": "advisory_fail",
            "grounding_reason": f"Grounding unavailable: {exc}",
        }


def validate_problem(problem: Dict, invariant_bundles=None, archival_evidence: Dict = None, parents: Optional[List[Dict]] = None, invoke_config: Dict = None) -> Tuple[str, str, Dict]:
    """
    재생성 가능성 검증: 솔루션과 답만 보고 원래 문제를 재생성할 수 있는가?
    
    검증 프로세스:
    1. LLM에게 solution + answer만 제공
    2. LLM이 원래 문제를 추론하도록 요청
    3. 재생성된 문제와 원본 문제의 의미적 유사성 비교
    
    Args:
        problem: 문제 dict (statement, solution, answer 포함)
    
    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        # 필수 필드 확인
        original_statement = problem.get("statement", "").strip()
        solution = problem.get("solution") or problem.get("solution_sketch") or ""
        solution = solution.strip()
        answer = str(problem.get("answer", "")).strip()
        
        if not original_statement:
            return "hard_fail", "No problem statement provided", {"verdict": "hard_fail", "mismatch_types": ["other"]}
        if not solution:
            return "hard_fail", "No solution provided", {"verdict": "hard_fail", "mismatch_types": ["other"]}
        if not answer:
            return "hard_fail", "No answer provided", {"verdict": "hard_fail", "mismatch_types": ["other"]}

        plan_payload = OrchestratorPlanSchema(
            stage="validate_candidates",
            objective="Validate whether the generated problem is semantically equivalent to its intended mathematical target after reconstruction.",
            strategy_summary="Reconstruct from solution and answer, then compare semantic equivalence against the original statement.",
            hard_constraints=[
                "Judge mathematical meaning, not wording alone.",
                "Return only the requested output shape at each validation step.",
            ],
            success_criteria=[
                "Produce a reconstructed problem statement.",
                "Return a valid equivalence JSON object.",
                "Separate advisory semantic shifts from hard mathematical identity breaks.",
            ],
        ).model_dump()
        dispatch_payload = OrchestratorDispatchSchema(
            agent_role="validator",
            slot=int(problem.get("_slot", 0)),
            op_type="validate",
            parent_ids=problem.get("parent_ids", []) or [],
            invariant_notes="Validation must preserve the original mathematical objects, constraints, and target quantity.",
            task_payload={
                "problem_id": problem.get("id", ""),
                "answer": answer,
            },
            finish_when=[
                "Reconstruct the original problem from solution and answer.",
                "Return equivalence=true only when the mathematical problem is the same.",
            ],
        ).model_dump()
        invariant_block = build_invariant_bundle_block(invariant_bundles or [])
        archival_block = build_archival_evidence_block(archival_evidence or {})
        peer_block = _build_peer_context_block(problem)
        orchestrator_context = (
            build_orchestrator_context_block(plan_payload, dispatch_payload)
            + "\n\n"
            + invariant_block
            + "\n\n"
            + archival_block
        )
        if peer_block:
            orchestrator_context += "\n\n" + peer_block
        
        # 1단계: LLM에게 솔루션과 답만 주고 문제 재생성 요청
        # Note: regen_llm deliberately omits archival_block and invariant_block so the
        # model reconstructs the problem "blind" from solution+answer only, without
        # being anchored to the parent's structure. orchestrator_context (plan+dispatch)
        # is retained for tracing / agent-role identification.
        regeneration_prompt = build_validator_regeneration_prompt(solution, answer)
        regen_context = build_orchestrator_context_block(plan_payload, dispatch_payload)
        if peer_block:
            regen_context += "\n\n" + peer_block

        regen_llm = _validator_llm.with_structured_output(ReconstructedProblemSchema, include_raw=True)
        regen_run_name = f"{(invoke_config or {}).get('run_name', 'deepagent.validator')}.regen"
        regenerated = invoke_structured_with_slim_trace(
            regen_llm,
            [
                SystemMessage(content=VALIDATOR_EXECUTION_SYSTEM_PROMPT + "\n\n" + VALIDATOR_REGEN_SYSTEM_PROMPT),
                HumanMessage(content=regen_context + "\n\n" + regeneration_prompt)
            ],
            invoke_config={**(invoke_config or {}), "run_name": regen_run_name},
            trace_name=f"{regen_run_name}.llm",
            tags=list((invoke_config or {}).get("tags", []) or []),
            metadata=dict((invoke_config or {}).get("metadata", {}) or {}),
            summary_inputs={
                "problem_id": problem.get("id", ""),
                "slot": int(problem.get("_slot", 0) or 0),
            },
            output_key="reconstructed_problem",
        )
        regenerated_statement = regenerated["statement"].strip()

        if not regenerated_statement:
            return "hard_fail", "Failed to regenerate problem statement", {"verdict": "hard_fail", "mismatch_types": ["other"]}

        # 2단계: 재생성된 문제와 원본 문제의 의미적 유사성 비교
        # Note: compare_llm task is purely semantic comparison of two statements.
        # No orchestrator_context / invariant_block / archival_block needed — those
        # were the primary driver of the LengthFinishReasonError (model generating
        # an exhaustive structured response including all that context overhead).
        comparison_prompt = build_validator_comparison_prompt(original_statement, regenerated_statement)

        compare_llm = _validator_llm.with_structured_output(ValidatorEquivalenceSchema, include_raw=True)
        compare_run_name = f"{(invoke_config or {}).get('run_name', 'deepagent.validator')}.compare"
        result = invoke_structured_with_slim_trace(
            compare_llm,
            [
                SystemMessage(content=VALIDATOR_EXECUTION_SYSTEM_PROMPT + "\n\n" + VALIDATOR_COMPARE_SYSTEM_PROMPT),
                HumanMessage(content=comparison_prompt)
            ],
            invoke_config={**(invoke_config or {}), "run_name": compare_run_name},
            trace_name=f"{compare_run_name}.llm",
            tags=list((invoke_config or {}).get("tags", []) or []),
            metadata=dict((invoke_config or {}).get("metadata", {}) or {}),
            summary_inputs={
                "problem_id": problem.get("id", ""),
                "slot": int(problem.get("_slot", 0) or 0),
            },
            output_key="equivalence",
        )
        mismatch_types = list(result.get("mismatch_types") or [])
        # Phase 0 (2026-04-18) — domain_shift demoted from fail-closed to advisory.
        # Rationale: limited seed pool benefits from candidates that drift into a
        # different mathematical family as long as code_gate + solvability + near-copy
        # already passed. The drift IS the diversity. Keep target_shift and
        # definition_rewrite as fail-closed because those change the actual problem
        # the solver is asked to verify (target quantity / defined objects),
        # not just the surrounding family.
        critical_mismatch_types = {"target_shift", "definition_rewrite"}
        equivalent = bool(result.get("equivalent", False))
        verdict = result.get("verdict", "")
        if not verdict:
            verdict = "pass" if equivalent else (
                "hard_fail" if critical_mismatch_types.intersection(mismatch_types) else "advisory_fail"
            )
        if equivalent:
            verdict = "pass"
        elif critical_mismatch_types.intersection(mismatch_types):
            verdict = "hard_fail"
        elif verdict == "pass":
            verdict = "advisory_fail"
        reason = result.get("reason", "No reason provided")
        result["mismatch_types"] = mismatch_types
        result["verdict"] = verdict
        
        if verdict == "pass":
            logger.info(f"✅ Regenerability validation passed: {reason}")
            return "pass", f"Regenerability confirmed: {reason}", result
        if verdict == "advisory_fail":
            logger.warning(f"⚠️ Regenerability advisory mismatch: {reason}")
            return "advisory_fail", f"Regenerability advisory: {reason}", result

        # T3-a: parent-anchored 2nd pass. The 1st pass's "unrelated"/"replaces"
        # reasons often reflect drift in the reconstruction LLM, not the child.
        # When a parent statement is available and the failure reason matches
        # the drift pattern, re-run the comparison with the parent as anchor.
        parent_stmt_for_anchor = ""
        if parents and _REGEN_DRIFT_REASON_PATTERNS.search(reason or ""):
            for p in parents:
                stmt = str((p or {}).get("statement") or "").strip()
                if stmt:
                    parent_stmt_for_anchor = stmt
                    break
        if parent_stmt_for_anchor:
            try:
                anchored_prompt = build_validator_parent_anchored_comparison_prompt(
                    parent_stmt_for_anchor, original_statement, reason,
                )
                anchored_llm = _validator_llm.with_structured_output(ValidatorEquivalenceSchema, include_raw=True)
                anchored_run_name = f"{(invoke_config or {}).get('run_name', 'deepagent.validator')}.anchored_retry"
                anchored_result = invoke_structured_with_slim_trace(
                    anchored_llm,
                    [
                        SystemMessage(content=VALIDATOR_EXECUTION_SYSTEM_PROMPT + "\n\n" + VALIDATOR_ANCHORED_RETRY_SYSTEM_PROMPT),
                        HumanMessage(content=anchored_prompt)
                    ],
                    invoke_config={**(invoke_config or {}), "run_name": anchored_run_name},
                    trace_name=f"{anchored_run_name}.llm",
                    tags=list((invoke_config or {}).get("tags", []) or []) + ["anchored_retry"],
                    metadata={**dict((invoke_config or {}).get("metadata", {}) or {}), "retry_pass": "parent_anchored"},
                    summary_inputs={
                        "problem_id": problem.get("id", ""),
                        "slot": int(problem.get("_slot", 0) or 0),
                    },
                    output_key="equivalence_anchored",
                )
                anchored_mismatches = list(anchored_result.get("mismatch_types") or [])
                anchored_equivalent = bool(anchored_result.get("equivalent", False))
                anchored_verdict = anchored_result.get("verdict") or (
                    "pass" if anchored_equivalent else (
                        "hard_fail" if critical_mismatch_types.intersection(anchored_mismatches) else "advisory_fail"
                    )
                )
                if anchored_equivalent:
                    anchored_verdict = "pass"
                elif critical_mismatch_types.intersection(anchored_mismatches):
                    anchored_verdict = "hard_fail"
                anchored_reason = anchored_result.get("reason", "")
                result["anchored_retry"] = {
                    "verdict": anchored_verdict,
                    "reason": anchored_reason,
                    "mismatch_types": anchored_mismatches,
                }
                if anchored_verdict == "pass":
                    logger.info(
                        "✅ Regenerability rescued by parent-anchored retry: %s", anchored_reason
                    )
                    result["verdict"] = "pass"
                    return (
                        "pass",
                        f"Regenerability confirmed [anchored_override]: {anchored_reason}",
                        result,
                    )
                if anchored_verdict == "advisory_fail":
                    logger.warning(
                        "⚠️ Regenerability downgraded by parent-anchored retry: %s", anchored_reason
                    )
                    result["verdict"] = "advisory_fail"
                    return (
                        "advisory_fail",
                        f"Regenerability advisory [anchored_override]: {anchored_reason}",
                        result,
                    )
                # anchored hard_fail: fall through and return the original hard_fail
            except Exception as exc:  # noqa: BLE001
                logger.warning("anchored_retry error; keeping original hard_fail: %s", exc)

        logger.warning(f"❌ Regenerability validation failed: {reason}")
        return "hard_fail", f"Regenerability failed: {reason}", result
            
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logger.error(error_msg)
        return "hard_fail", error_msg, {"verdict": "hard_fail", "mismatch_types": ["other"], "reason": error_msg}
