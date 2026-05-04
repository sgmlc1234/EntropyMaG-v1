"""Regen planner — orchestrator's retry-route stage.

After a validation round produces failed_problems, this module runs ONE LLM
call over the failed batch and emits a per-slot decision about whether each
slot should (a) be re-sent to the researcher for fresh idea mining, (b) go
through the deterministic repair hotfixes, (c) drop to easier mode, or (d) be
abandoned so the elite-backfill path can rescue throughput.

The orchestrator owns this decision — there is no hardcoded rule mapping
failure_type → route; instead, structured failure snapshots are fed to an LLM
along with the available research-refetch budget, and the LLM emits a
RegenPlanSchema envelope that ``regenerate_failed_node`` applies to the retry
items.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_llm_config
from deepagent.tracing import invoke_structured_with_slim_trace
from prompts import (
    REGEN_PLAN_SYSTEM_PROMPT,
    RegenPlanDispatchArgs,
    RegenPlanSchema,
    SlotFailureSummarySchema,
    build_regen_plan_prompt,
)

logger = logging.getLogger("deep_regen_planner")

_cfg = get_llm_config("regen_planner")
_planner_llm = ChatOpenAI(
    model=_cfg["model"],
    temperature=0.1,
    max_tokens=_cfg.get("max_tokens", 40000),
    timeout=_cfg.get("timeout", 120),
    max_retries=_cfg.get("max_retries", 3),
    api_key=_cfg["api_key"],
    base_url=_cfg["base_url"],
).with_structured_output(RegenPlanSchema, include_raw=True)


_PROBLEMATIC_QUERY_VERBS = re.compile(
    r"\b(determine|find|compute|solve|prove|maximi[sz]e|minimi[sz]e|what\s+is|how\s+many|evaluate|calculate|show\s+that)\b",
    re.IGNORECASE,
)

_CANONICAL_REPAIR_STRATEGIES = (
    "full_regenerate",
    "code_only_hotfix",
    "target_only_hotfix",
    "statement_domain_hotfix",
)


def _truncate(text: str, limit: int) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if len(t) <= limit:
        return t
    return t[: limit - 1].rstrip() + "\u2026"


_MECHANICAL_FAILURE_TYPES = {
    "missing_required_fields",
    "code_gate_failure",
    "sham_code_failure",
    "statement_code_inconsistency",
    "statement_solution_inconsistency",
    "solvability_failure",
    "target_quantity_failure",
    "relation_guard_failure",
}

_IDEA_SCARCITY_FAILURE_TYPES = {
    "novelty_collapse_failure",
    "exploration_failure",
    "invariant_failure",
    "regenerability_failure",
    "ground_reject",
    "ground_rescope",
}


def _canonical_signature(sig: str, failure_type: str) -> str:
    """Collapse a reason_signature to a drift-resistant canonical key.

    `_reason_signature` returns named tags ("novelty_collapse", "quality_gate",
    ...) for recognized patterns but falls through to `text[:120]` for the
    large residual bucket — where a 1–3 character surface change (a slash vs
    space, a different word choice) defeats exact-string equality in
    `_persistent_failure`. For comparison purposes, prefer the failure_type
    label (which already generalizes across drift variants), then fall back
    to digit-masked, first-8-token form.
    """
    if failure_type and failure_type not in ("", "other"):
        return failure_type
    text = re.sub(r"\d+", "N", (sig or "").lower())
    tokens = re.sub(r"[^a-z ]", " ", text).split()[:8]
    return " ".join(tokens)


def _persistent_failure(slot: Dict[str, Any]) -> bool:
    """True iff the last 3 reason_signatures for this slot collapse to one key.

    Signals that repeated retries are converging on the same failure mode
    rather than exploring the failure space — i.e. further repair is unlikely
    to succeed without a structural change. Uses `_canonical_signature` so
    cosmetic drift in the free-text part of a regenerability reason does not
    defeat the check.
    """
    recent = list(slot.get("recent_failure_signatures") or [])
    recent_types = list(slot.get("recent_failure_types") or [])
    # Pad failure_types with empty strings so indices line up with signatures
    # even if the upstream only populated the more recent entries.
    while len(recent_types) < len(recent):
        recent_types.insert(0, "")
    pairs = [(s, t) for s, t in zip(recent[-3:], recent_types[-3:]) if s]
    if len(pairs) < 3:
        return False
    canonical = {_canonical_signature(s, t) for s, t in pairs}
    return len(canonical) == 1


def _classify_route_fallback(slot: Dict[str, Any], *, max_slot_regen_attempts: int = 2) -> str:
    """Deterministic fallback mirroring regen_plan.md §Tiebreaker priority.

    Evaluated top-to-bottom; first match wins:
      0. attempts ≥ max_slot_regen_attempts AND last 3 signatures identical → giveup
      1. budget_exhausted → escalate_easier if attempts ≥ 2 else direct_repair
      2. repeated_signature AND idea-scarcity failure → research_and_regenerate
      3. mechanical failure → direct_repair
      4. attempts ≥ 2 AND failure_type ends in "_failure" → escalate_easier
      5. otherwise → direct_repair
    """
    attempts = int(slot.get("attempts", 0) or 0)
    failure_type = (slot.get("failure_type") or "").lower()
    refetch_count = int(slot.get("research_refetch_count", 0) or 0)
    refetch_budget = int(slot.get("research_refetch_budget", 1) or 1)
    repeated = bool(slot.get("repeated_signature"))
    op_type = str(slot.get("op_type") or "").lower()

    # Rule -1: survivor op_type cannot be repaired; _run_repair_item raises on it.
    # Route to giveup so elite_backfill rescues the slot immediately.
    if op_type == "survivor":
        return "giveup"

    # Rule 0: hard cap exceeded with persistent same-signature failure → giveup.
    # Elite backfill will rescue the slot. Do not loop further.
    if attempts >= max_slot_regen_attempts and _persistent_failure(slot):
        return "giveup"
    # Rule 1: budget exhausted
    if refetch_count >= refetch_budget:
        return "escalate_easier" if attempts >= 2 else "direct_repair"
    # Rule 2: idea scarcity with repeated signature and budget remaining
    if repeated and failure_type in _IDEA_SCARCITY_FAILURE_TYPES:
        return "research_and_regenerate"
    # Rule 3: mechanical failures route to repair
    if failure_type in _MECHANICAL_FAILURE_TYPES:
        return "direct_repair"
    # Rule 4: repeated hard-mode failures escalate
    if attempts >= 2 and failure_type.endswith("_failure"):
        return "escalate_easier"
    # Rule 5: default
    return "direct_repair"


def _repair_brief_fallback(summary: Dict[str, Any]) -> str:
    """Produce a deterministic repair_brief when the LLM call is unavailable.

    Combines the most recent failure types/signatures and prior decisions into
    a 'Pattern / Tried / Try instead' synthesis line. Used as the fallback
    value for direct_repair / escalate_easier decisions emitted by
    `_classify_route_fallback`. Empty for research_and_regenerate and giveup.
    """
    recent_types = list(summary.get("recent_failure_types") or [])[-3:]
    recent_sigs = list(summary.get("recent_failure_signatures") or [])[-3:]
    prior_decisions = list(summary.get("prior_decisions") or [])[-3:]
    parts = []
    if recent_types:
        parts.append(f"Pattern: {recent_types}")
    if recent_sigs:
        parts.append(f"signatures={recent_sigs}")
    if prior_decisions:
        parts.append(f"Tried: {prior_decisions}")
    parts.append("Try instead: a structurally different fix from prior attempts.")
    text = " ".join(parts)
    return _truncate(text, 480)


def _technique_fallback_from_failure(summary: Dict[str, Any]) -> str:
    """Produce a policy-compliant technique-only research_focus for fallback.

    Uses the variation axis and failure signature rather than any parent or
    child statement tokens so the resulting query cannot drift back into
    problem-restatement territory.
    """
    axis = _truncate(str(summary.get("variation_axis") or ""), 48)
    failure_type = _truncate(str(summary.get("failure_type") or "generic_failure"), 48)
    phrases = []
    if axis:
        phrases.append(f"alternative techniques along {axis}")
    phrases.append(f"deep structural composition avoiding {failure_type}")
    if summary.get("op_type") == "crossover":
        phrases.append("cross-family bridging invariants")
    else:
        phrases.append("single-family deepening transforms")
    combined = " || ".join(phrases[:3])
    # Defensive guard — sanitize against any residual problem-statement verb.
    combined = _PROBLEMATIC_QUERY_VERBS.sub("", combined).strip(" .,-")
    return combined[:220]


def _fallback_plan(
    failed_slot_payload: List[Dict[str, Any]],
    *,
    max_slot_regen_attempts: int = 2,
) -> Dict[str, Any]:
    """Construct a RegenPlanSchema payload when the LLM call fails."""
    decisions = []
    for summary in failed_slot_payload:
        decision = _classify_route_fallback(summary, max_slot_regen_attempts=max_slot_regen_attempts)
        research_focus = (
            _technique_fallback_from_failure(summary)
            if decision == "research_and_regenerate"
            else ""
        )
        attempts = int(summary.get("attempts", 0) or 0)
        repair_brief = (
            _repair_brief_fallback(summary)
            if decision in {"direct_repair", "escalate_easier"} and attempts >= 3
            else ""
        )
        decisions.append(
            {
                "slot": int(summary.get("slot", 0) or 0),
                "decision": decision,
                "research_focus_override": research_focus,
                "repair_strategy_override": "",
                "retry_feedback_summary": _truncate(summary.get("latest_failure_summary", ""), 280)
                or f"Repeat of {summary.get('failure_type', 'unknown')} failure.",
                "repair_brief": repair_brief,
                "rationale": (
                    "Fallback heuristic: idea-scarcity → research; else mechanical → repair."
                ),
            }
        )
    return {
        "per_slot_decisions": decisions,
        "overall_rationale": "Fallback heuristic used because the orchestrator LLM call was unavailable.",
    }


def _sanitize_decision(
    decision: Dict[str, Any],
    slot_summary: Optional[Dict[str, Any]] = None,
    *,
    max_slot_regen_attempts: int = 2,
) -> Dict[str, Any]:
    """Enforce hard invariants on each decision regardless of what the LLM emitted.

    Mirrors regen_plan.md §Tiebreaker priority:
      * Persistent same-signature failures past the attempts cap are force-routed
        to giveup (Rule 0).
      * Slots that already consumed their research refetch budget cannot be
        routed to research_and_regenerate. Downgrade path: escalate_easier if
        attempts ≥ 2, else direct_repair.
      * research_focus_override is required iff decision == research_and_regenerate.
      * Problem-statement verbs in research_focus_override trip a warning and the
        value is cleared (the research fallback path will then degrade cleanly).
    """
    slot_summary = slot_summary or {}
    slot = int(decision.get("slot", 0) or 0)
    route = decision.get("decision", "direct_repair")
    override = (decision.get("research_focus_override") or "").strip()

    refetch_count = int(slot_summary.get("research_refetch_count", 0) or 0)
    refetch_budget = int(slot_summary.get("research_refetch_budget", 1) or 1)
    attempts = int(slot_summary.get("attempts", 0) or 0)
    budget_exhausted = refetch_count >= refetch_budget
    op_type = str(slot_summary.get("op_type") or "").lower()

    # Rule -1: survivor op_type is not repairable. Force giveup so elite_backfill
    # rescues the slot; _run_repair_item would otherwise raise.
    if op_type == "survivor" and route != "giveup":
        logger.warning(
            "regen_planner: slot %s force-routed to giveup (op_type=survivor); LLM proposed %r.",
            slot,
            route,
        )
        route = "giveup"
        override = ""

    # Rule 0 enforcement (force-promote): force giveup on persistent same-signature
    # failures past the cap, regardless of LLM's decision.
    if (
        route != "giveup"
        and attempts >= max_slot_regen_attempts
        and _persistent_failure(slot_summary)
    ):
        logger.warning(
            "regen_planner: slot %s force-routed to giveup — attempts=%s, "
            "last 3 signatures identical; LLM proposed %r.",
            slot,
            attempts,
            route,
        )
        route = "giveup"
        override = ""

    # Rule 0 enforcement (downgrade): if LLM proposed giveup but the slot has
    # not actually met Rule 0 conditions (attempts cap + persistent signature),
    # downgrade to direct_repair so the slot gets a real retry instead of
    # being prematurely abandoned. Without this guard, the LLM occasionally
    # hallucinates "Mandatory giveup per Rule 0" rationale on a first-attempt
    # slot, which collapses throughput.
    if route == "giveup" and op_type != "survivor" and not (
        attempts >= max_slot_regen_attempts and _persistent_failure(slot_summary)
    ):
        logger.warning(
            "regen_planner: slot %s LLM proposed giveup but Rule 0 conditions not met "
            "(attempts=%s, max_cap=%s, persistent=%s); downgrading to direct_repair.",
            slot,
            attempts,
            max_slot_regen_attempts,
            _persistent_failure(slot_summary),
        )
        route = "direct_repair"

    if route == "research_and_regenerate" and budget_exhausted:
        failure_type = (slot_summary.get("failure_type") or "").lower()
        # T2-c: if the slot is stuck in an idea-scarcity loop (regenerability drift,
        # exploration_failure, etc.) AND the signature has converged (_persistent_failure),
        # further repair is architecturally doomed — route to giveup so elite_backfill
        # rescues the slot instead of burning 1-2 more attempts on direct_repair.
        if failure_type in _IDEA_SCARCITY_FAILURE_TYPES and _persistent_failure(slot_summary):
            downgrade = "giveup"
        else:
            downgrade = "escalate_easier" if attempts >= 2 else "direct_repair"
        logger.warning(
            "regen_planner: slot %s exceeded research_refetch_budget but LLM proposed research; "
            "downgrading to %s (attempts=%s, failure_type=%s).",
            slot,
            downgrade,
            attempts,
            failure_type,
        )
        route = downgrade
        override = ""

    if route == "research_and_regenerate" and _PROBLEMATIC_QUERY_VERBS.search(override):
        logger.warning(
            "regen_planner: slot %s research_focus_override contained a problem-statement verb; clearing. original=%r",
            slot,
            override[:200],
        )
        override = ""

    if route != "research_and_regenerate":
        override = ""

    # T2-d: canonicalize repair_strategy_override. LLMs occasionally emit free-text
    # tokens (e.g. "remove_numpy_dependency") that silently degrade to full_regenerate
    # in `_run_repair_item` — here we map fuzzy variants back to the canonical set
    # and drop truly unknown values so deterministic routing can take over.
    raw_strategy_override = (decision.get("repair_strategy_override") or "").strip()
    strategy_override = ""
    if raw_strategy_override:
        normalized = re.sub(r"[\s\-]+", "_", raw_strategy_override.lower()).strip("_")
        if normalized in _CANONICAL_REPAIR_STRATEGIES:
            strategy_override = normalized
        elif raw_strategy_override.lower() in _CANONICAL_REPAIR_STRATEGIES:
            strategy_override = raw_strategy_override.lower()
        else:
            logger.warning(
                "regen_planner: slot %s dropped non-canonical repair_strategy_override=%r; "
                "will fall back to deterministic strategy.",
                slot,
                raw_strategy_override[:80],
            )

    # T2-b: prior_repair_strategies cycling. If the effective strategy for this
    # attempt (LLM override or deterministic fallback) was already tried, pick
    # the first untried canonical; if ALL canonical strategies have been tried,
    # promote route to giveup so elite_backfill rescues the slot.
    if route in {"direct_repair", "escalate_easier"}:
        prior_strategies = list(slot_summary.get("prior_repair_strategies") or [])
        if strategy_override:
            effective = strategy_override
        else:
            try:
                from deepagent.graph.helpers import _repair_strategy_for_failure
                effective = _repair_strategy_for_failure(
                    slot_summary.get("failure_type", "") or "",
                    slot_summary.get("recent_failure_signatures", [""])[-1] if slot_summary.get("recent_failure_signatures") else "",
                )
            except Exception:  # noqa: BLE001
                effective = "full_regenerate"
        tried = {s for s in prior_strategies if s in _CANONICAL_REPAIR_STRATEGIES}
        if effective in tried:
            untried = [s for s in _CANONICAL_REPAIR_STRATEGIES if s not in tried]
            if untried:
                logger.info(
                    "regen_planner: slot %s cycling repair strategy — tried=%s, picking %s.",
                    slot, sorted(tried), untried[0],
                )
                strategy_override = untried[0]
            else:
                logger.warning(
                    "regen_planner: slot %s all 4 canonical repair strategies already tried; "
                    "promoting route %s -> giveup.",
                    slot, route,
                )
                route = "giveup"
                override = ""
                strategy_override = ""
    else:
        # Non-repair routes ignore repair_strategy_override.
        strategy_override = ""

    # repair_brief is required when decision in {direct_repair, escalate_easier}.
    # Phase-1 Fix-#1 (2026-04-18): for regenerability_failure specifically, ALWAYS
    # emit a parent-anchored brief — drop the `attempts >= 3` gate. The post-mortem
    # of the 18 failed-repair slots showed every regenerability failure entered
    # repair with no brief at all (attempts <= 3), so the worker had nothing
    # but free-text validation_feedback to act on, and predictably drifted
    # again. The parent-anchored skeleton pins the parent_id + variation_axis
    # in the brief so the worker cannot abandon the family on retry.
    repair_brief = (decision.get("repair_brief") or "").strip()
    if route in {"direct_repair", "escalate_easier"}:
        failure_type = (slot_summary.get("failure_type") or "").lower()
        latest_sig = (slot_summary.get("latest_failure_summary") or "").lower()
        # Phase-1 Fix-#2 (2026-04-18): signature-specific deterministic briefs
        # for the three postprocess patterns that the failure_type-only path
        # historically mis-routed. These run BEFORE the regenerability anchor
        # so they short-circuit when the signature actually matches.
        signature_brief = ""
        if "syntax_error" in latest_sig and "line 1" in latest_sig:
            signature_brief = (
                "Code field has a line-1 syntax error. Re-emit the whole `code` field as exactly: "
                "```python\\n<imports>\\n<solve_function>\\n<print_single_line_answer>\\n``` "
                "with NO markdown fence, NO leading blank line, NO commentary. Verify the first character is a valid Python token."
            )
        elif "could not extract canonical answer" in latest_sig:
            signature_brief = (
                "Print exactly ONE line containing only the answer (integer, fraction, or short literal). "
                "Remove all debug strings, label prefixes (`Answer:`, `Result =`), tuples, and multi-value prints. "
                "The validator parses the LAST stdout line as the canonical answer."
            )
        elif "not in" in latest_sig and "statement" in latest_sig:
            signature_brief = (
                "Code references a constant or variable that the statement never declares. "
                "Either (a) update the statement to declare it explicitly with the value used in code, or (b) restructure code to derive the value from quantities the statement DOES declare. Do not leave any code-only constants."
            )
        elif "constraint system is inconsistent" in latest_sig:
            signature_brief = (
                "The stated constraints have NO simultaneous solution. Either weaken a constraint, drop a redundant one, or reframe the target so the system is satisfiable. Do not silently change the answer."
            )
        if failure_type == "regenerability_failure":
            parent_ids = slot_summary.get("parent_ids") or []
            variation_axis = (slot_summary.get("variation_axis") or "").strip()
            op_type_summary = (slot_summary.get("op_type") or "mutation").lower()
            anchor_lines: List[str] = []
            if parent_ids:
                anchor_lines.append(
                    f"Repair this {op_type_summary} of parent(s) {', '.join(parent_ids[:2])}."
                )
            anchor_lines.append(
                "MUST preserve the parent's named definitions, core relations, and target-quantity kind."
            )
            if variation_axis:
                anchor_lines.append(
                    f"Allowed variation: {variation_axis[:160]}. Stay strictly within this axis; do NOT introduce a different problem family."
                )
            anchor_lines.append(
                "DO NOT replace the parent's task type with a structurally unrelated one (e.g. coding theory \u2192 graph coloring)."
            )
            if not repair_brief:
                repair_brief = " ".join(anchor_lines)
            else:
                # Prepend the anchor so it dominates downstream attention.
                repair_brief = " ".join(anchor_lines) + " " + repair_brief
        elif signature_brief:
            # Phase-1 Fix-#2: prepend signature brief regardless of attempts
            # count — these patterns are deterministic enough that the worker
            # benefits from the directive on the first repair attempt.
            repair_brief = (signature_brief + " " + repair_brief).strip() if repair_brief else signature_brief
        elif not repair_brief and attempts >= 3:
            repair_brief = _repair_brief_fallback(slot_summary)
    else:
        repair_brief = ""

    return {
        "slot": slot,
        "decision": route,
        "research_focus_override": override[:320],
        "repair_strategy_override": _truncate(strategy_override, 80),
        "retry_feedback_summary": _truncate(decision.get("retry_feedback_summary") or "", 400),
        "repair_brief": _truncate(repair_brief, 500),
        "rationale": _truncate(decision.get("rationale") or "", 240),
    }


def plan_one_slot_regen(
    slot_summary: Dict[str, Any],
    *,
    generation_count: int,
    retry_round: int,
    research_refetch_budget: int = 1,
    max_slot_regen_attempts: int = 2,
    invoke_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-slot regen planning — single-slot LLM call.

    Phase E: replaces batch `plan_regeneration` for the cases where each
    slot's regen decision should be traced INSIDE the slot's own
    `generator.operation.slot_{slot}` span (the caller is expected to set
    `tracing_context(parent=slot_run)` before calling).

    Returns one validated `RegenSlotDecisionSchema` dict (the same shape
    as one entry of `plan_regeneration`'s `per_slot_decisions`).
    """
    plan = plan_regeneration(
        [slot_summary],
        generation_count=generation_count,
        retry_round=retry_round,
        research_refetch_budget=research_refetch_budget,
        max_slot_regen_attempts=max_slot_regen_attempts,
        invoke_config=invoke_config,
    )
    decisions = plan.get("per_slot_decisions") or []
    if decisions:
        return decisions[0]
    # Fallback if LLM produced no decision: derive deterministic one.
    return _fallback_plan([slot_summary], max_slot_regen_attempts=max_slot_regen_attempts)["per_slot_decisions"][0]


def plan_regeneration(
    failed_slot_summaries: List[Dict[str, Any]],
    *,
    generation_count: int,
    retry_round: int,
    research_refetch_budget: int = 1,
    max_slot_regen_attempts: int = 2,
    memory_summary_block: str = "",
    invoke_config: Optional[Dict[str, Any]] = None,
    dispatch_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one orchestrator LLM call that decides the retry route for each slot.

    Returns a validated RegenPlanSchema dict. Always returns a plan with one
    decision per input slot; missing decisions are back-filled via the
    deterministic fallback and decisions that violate hard invariants are
    sanitized.
    """
    if not failed_slot_summaries:
        return {"per_slot_decisions": [], "overall_rationale": "No failed slots."}

    # Validate summaries through the schema so the prompt always sees
    # well-formed entries.
    validated_summaries = [
        SlotFailureSummarySchema.model_validate(summary).model_dump()
        for summary in failed_slot_summaries
    ]
    if dispatch_args is None:
        dispatch_args = RegenPlanDispatchArgs.model_validate(
            {
                "generation_count": int(generation_count),
                "retry_round": int(retry_round),
                "research_refetch_budget": int(research_refetch_budget),
                "max_slot_regen_attempts": int(max_slot_regen_attempts),
                "failed_slots": validated_summaries,
            }
        ).model_dump()

    # When all slots have exhausted their research refetch budget, the LLM
    # cannot produce any decision different from the deterministic fallback
    # (research_and_regenerate is always downgraded by _sanitize_decision).
    # Skip the LLM call entirely to avoid ~2.4s of wasted latency per cycle.
    all_budgets_exhausted = all(
        int(s.get("research_refetch_count", 0)) >= int(s.get("research_refetch_budget", research_refetch_budget))
        for s in validated_summaries
    )
    if all_budgets_exhausted:
        logger.info(
            "regen_planner: all %d slot(s) have exhausted research budget — using deterministic fallback.",
            len(validated_summaries),
        )
        raw_plan = _fallback_plan(validated_summaries, max_slot_regen_attempts=max_slot_regen_attempts)
        per_slot = {int(d.get("slot", 0) or 0): d for d in (raw_plan.get("per_slot_decisions") or [])}
        merged_decisions = [
            _sanitize_decision(
                per_slot.get(int(s.get("slot", 0) or 0), _fallback_plan([s], max_slot_regen_attempts=max_slot_regen_attempts)["per_slot_decisions"][0]),
                s,
                max_slot_regen_attempts=max_slot_regen_attempts,
            )
            for s in validated_summaries
        ]
        return RegenPlanSchema.model_validate({
            "per_slot_decisions": merged_decisions,
            "overall_rationale": "Deterministic fallback: all research budgets exhausted.",
        }).model_dump()

    prompt = build_regen_plan_prompt(
        generation_count=int(generation_count),
        retry_round=int(retry_round),
        research_refetch_budget=int(research_refetch_budget),
        max_slot_regen_attempts=int(max_slot_regen_attempts),
        failed_slots_block=json.dumps(validated_summaries, ensure_ascii=False, indent=2),
        memory_summary_block=memory_summary_block or "",
    )

    try:
        raw_plan = invoke_structured_with_slim_trace(
            _planner_llm,
            [
                SystemMessage(content=REGEN_PLAN_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            invoke_config=invoke_config or {"tags": ["deepagent", "regen_planner"]},
            trace_name=(invoke_config or {}).get("run_name") or "deepagent.regen_planner.llm",
            tags=["deepagent", "regen_planner", "structured"],
            metadata={
                "generation_count": int(generation_count),
                "retry_round": int(retry_round),
                "slot_count": len(validated_summaries),
            },
            summary_inputs={
                "generation_count": int(generation_count),
                "retry_round": int(retry_round),
                "slots": [int(s["slot"]) for s in validated_summaries],
                "budget_exhausted_slots": sorted(
                    int(s["slot"])
                    for s in validated_summaries
                    if int(s.get("research_refetch_count", 0)) >= int(s.get("research_refetch_budget", 1))
                ),
            },
            output_key="plan",
        )
    except Exception as exc:  # noqa: BLE001 — we always need to fall back to a deterministic plan
        logger.warning("regen_planner LLM call failed (%s); using deterministic fallback.", exc)
        raw_plan = _fallback_plan(validated_summaries, max_slot_regen_attempts=max_slot_regen_attempts)

    if not isinstance(raw_plan, dict):
        raw_plan = _fallback_plan(validated_summaries, max_slot_regen_attempts=max_slot_regen_attempts)

    per_slot = {int(d.get("slot", 0) or 0): d for d in (raw_plan.get("per_slot_decisions") or [])}
    # Backfill any missing slot with the deterministic fallback.
    merged_decisions: List[Dict[str, Any]] = []
    for summary in validated_summaries:
        slot = int(summary.get("slot", 0) or 0)
        if slot in per_slot:
            merged_decisions.append(
                _sanitize_decision(per_slot[slot], summary, max_slot_regen_attempts=max_slot_regen_attempts)
            )
        else:
            fallback_single = _fallback_plan([summary], max_slot_regen_attempts=max_slot_regen_attempts)["per_slot_decisions"][0]
            merged_decisions.append(
                _sanitize_decision(fallback_single, summary, max_slot_regen_attempts=max_slot_regen_attempts)
            )

    plan = {
        "per_slot_decisions": merged_decisions,
        "overall_rationale": _truncate(
            str(raw_plan.get("overall_rationale", "") or "Regen plan applied."),
            320,
        ),
    }
    # Run pydantic validation as a final safety net; this also enforces the
    # Literal choices on `decision` and the field-level constraints.
    return RegenPlanSchema.model_validate(plan).model_dump()
