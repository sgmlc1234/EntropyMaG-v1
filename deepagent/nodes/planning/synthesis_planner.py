"""Synthesis planner — orchestrator's second-stage plan (per-slot, pre-research).

The selector (plan_generation) picks op_type, parents, variation_axis, and seed_focus.
This module commits every non-survivor slot to the research-independent synthesis
decisions (composition pattern, parameter reuse policy, target/relation guards,
concept hooks) BEFORE research runs, and emits a research_focus that shapes the
researcher's query.

Downstream, ``nodes/planning/briefing.py`` overlays the research artifact
onto this plan deterministically (Phase D2a: no more LLM brief_validator
stage) to produce the dict consumed by the generator.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_llm_config
from deepagent.family_strategy import infer_family_signature
from deepagent.memory_bank import derive_family_policy
from deepagent.tracing import invoke_structured_with_slim_trace
from prompts import (
    GROUPED_SYNTHESIS_PLAN_SYSTEM_PROMPT,
    GroupedSynthesisPlanSchema,
    SynthesisPlanDispatchArgs,
    SynthesisPlanSchema,
    build_context_pack_block,
    build_invariant_bundle_block,
)

logger = logging.getLogger("deep_synthesis_planner")

_cfg = get_llm_config("synthesis_planner")
_grouped_planner_llm = ChatOpenAI(
    model=_cfg["model"],
    temperature=0.1,
    max_tokens=_cfg.get("max_tokens", 8000),
    timeout=_cfg.get("timeout", 120),
    max_retries=_cfg.get("max_retries", 3),
    api_key=_cfg["api_key"],
    base_url=_cfg["base_url"],
).with_structured_output(GroupedSynthesisPlanSchema, include_raw=True)


def _truncate(text: str, limit: int) -> str:
    if not isinstance(text, str):
        return ""
    trimmed = text.strip()
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[: limit - 1].rstrip() + "\u2026"


def _build_parent_body_block(parents: List[Dict[str, Any]]) -> str:
    if not parents:
        return "Parent bodies: (none provided)"
    chunks = []
    for parent in parents[:2]:
        pid = parent.get("id", "")
        statement = _truncate(parent.get("statement", ""), 700) or "(missing)"
        solution = _truncate(parent.get("solution", ""), 450) or "(missing)"
        answer = _truncate(str(parent.get("answer", "")), 120) or "(missing)"
        family = parent.get("family_signature", "") or infer_family_signature(parent) or "(unknown)"
        chunks.append(
            f"[{pid}] family={family}\n"
            f"  statement: {statement}\n"
            f"  answer: {answer}\n"
            f"  solution: {solution}"
        )
    return "Parent bodies (authoritative):\n" + "\n".join(chunks)


def _fallback_synthesis_plan(
    work_item: Dict[str, Any],
    context_pack: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context_pack = dict(context_pack or {})
    op_type = work_item.get("op_type", "mutation")
    if op_type == "survivor":
        op_type = "mutation"
    family_signatures = [
        card.get("family_signature", "")
        for card in (context_pack.get("lineage_context", []) or [])[:2]
        if card.get("family_signature")
    ]
    family_policy = derive_family_policy(family_signatures)
    preferred_pattern = (
        (context_pack.get("opportunity_context", {}) or {}).get("preferred_composition_pattern", "")
        or ("single_family_mutation" if op_type == "mutation" else "cross_family_bridge")
    )
    if preferred_pattern not in {
        "serial_pipeline",
        "same_system_new_parameters",
        "coupled_system_extension",
        "single_family_mutation",
        "cross_family_bridge",
    }:
        preferred_pattern = "single_family_mutation" if op_type == "mutation" else "cross_family_bridge"
    target_quantity_guard = "; ".join(
        bundle.get("target_quantity", "")
        for bundle in (work_item.get("invariant_bundles", []) or [])[:2]
        if bundle.get("target_quantity")
    ) or "Keep the parent target quantity explicit and unchanged."
    relation_guard = "Preserve exact equality, iff, and set-identity semantics from the parent."
    concept = (
        work_item.get("seed_focus")
        or ((context_pack.get("opportunity_context", {}) or {}).get("preferred_delta") or "")
        or "Activate the assigned variation_axis without drifting family."
    )
    pattern_to_avoid = "Shallow coefficient reshuffle or near-copy of the parent."
    # Prefer an explicit query_hint when the selector / orchestrator already
    # supplied a technique-oriented one. Otherwise treat the slot as
    # self-contained rather than emitting a free-text justification that could
    # be misread downstream as a research query.
    query_hint = (work_item.get("query_hint") or "").strip()
    research_required = bool(query_hint)
    return {
        "slot": int(work_item.get("slot", 0) or 0),
        "pair_id": work_item.get("pair_id", "") or "",
        "op_type": op_type,
        "target_quantity_guard": target_quantity_guard,
        "relation_guard": relation_guard,
        "concept_to_activate": concept,
        "pattern_to_avoid": pattern_to_avoid,
        "preferred_composition_pattern": preferred_pattern,
        "parameter_reuse_policy": family_policy["parameter_reuse_policy"],
        "deep_variant_requirement": family_policy["deep_variant_requirement"],
        "research_required": research_required,
        "research_focus": query_hint if research_required else "",
    }


def build_synthesis_plan_dispatch_args(
    work_item: Dict[str, Any],
    *,
    retry_feedback: str = "",
) -> Dict[str, Any]:
    """Project a work_item into a validated SynthesisPlanDispatchArgs payload.

    Raises pydantic ValidationError if required fields drift — this is the
    fail-fast boundary between orchestrator state and the synthesis planner.
    """
    op_type = work_item.get("op_type", "mutation")
    if op_type == "survivor":
        raise ValueError("synthesis_plan dispatch is not valid for survivor slots")
    mode = work_item.get("mode") or "hard"
    if mode == "carry":
        mode = "hard"
    args = SynthesisPlanDispatchArgs.model_validate(
        {
            "slot": int(work_item.get("slot", 0) or 0),
            "pair_id": str(work_item.get("pair_id", "") or ""),
            "op_type": op_type,
            "parent_ids": list(work_item.get("parent_ids", []) or []),
            "variation_axis": str(work_item.get("variation_axis", "") or ""),
            "seed_focus": str(work_item.get("seed_focus", "") or ""),
            "mode": mode,
            "retry_feedback": retry_feedback or "",
        }
    )
    return args.model_dump()


def build_grouped_synthesis_plan_prompt(
    items: List[Dict[str, Any]],
    *,
    op_type: str,
) -> str:
    """Build the human-turn prompt for the grouped (op-type-level) synthesis planner.

    Each slot in ``items`` gets its own context block (parents, invariant bundles,
    context pack). The footer restates the diversity requirement so the LLM cannot
    miss it.
    """
    slot_blocks = []
    for item in items:
        slot = int(item.get("slot", 0) or 0)
        pair_id = item.get("pair_id", "") or ""
        variation_axis = item.get("variation_axis", "") or ""
        seed_focus = item.get("seed_focus", "") or ""
        retry_feedback = item.get("validation_feedback") or item.get("constraint_guard_feedback") or ""

        parents = list(item.get("parents", []) or [])
        invariant_bundles = list(item.get("invariant_bundles", []) or [])
        context_pack = dict(item.get("context_pack") or {})

        parent_body_block = _build_parent_body_block(parents)
        invariant_block = build_invariant_bundle_block(invariant_bundles)
        context_pack_block = build_context_pack_block(context_pack, stage="briefing")

        retry_block = ""
        if retry_feedback:
            retry_block = f"\nPrior failed-attempt feedback:\n{retry_feedback}\n"

        slot_blocks.append(
            f"=== SLOT {slot} (pair_id={pair_id}, variation_axis={variation_axis}, seed_focus={seed_focus or '(unspecified)'}) ===\n"
            f"{parent_body_block}\n\n"
            f"{invariant_block}\n\n"
            f"{context_pack_block}"
            f"{retry_block}"
        )

    slots_section = "\n\n".join(slot_blocks)
    return f"""Plan synthesis for ALL {op_type} slots in this generation in a single portfolio-coordinated pass.

Slot count: {len(items)}

{slots_section}

Portfolio diversity requirement:
- Each slot MUST receive a DIFFERENT preferred_composition_pattern.
- Each slot MUST activate a DIFFERENT concept hook (concept_to_activate).
- research_focus values across slots MUST be distinct and non-overlapping.

Emit one SynthesisPlanSchema per slot (slot_plans list). Keep every field short but concrete and grounded in the parent body. Return JSON only.
"""


def generate_grouped_synthesis_plan(
    items: List[Dict[str, Any]],
    *,
    op_type: str,
    invoke_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Produce synthesis plans for ALL slots of one op_type in a single LLM call.

    Returns a list of ``SynthesisPlanSchema`` dicts, one per item, in the same
    order as ``items``.  Falls back to per-slot ``_fallback_synthesis_plan`` for
    any slot missing from the LLM response or on total LLM failure.
    """
    fallbacks: Dict[int, Dict[str, Any]] = {
        int(item.get("slot", 0) or 0): _fallback_synthesis_plan(item, item.get("context_pack"))
        for item in items
    }

    prompt = build_grouped_synthesis_plan_prompt(items, op_type=op_type)
    messages = [
        SystemMessage(content=GROUPED_SYNTHESIS_PLAN_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    summary_inputs = {
        "op_type": op_type,
        "slot_count": len(items),
        "slots": [int(item.get("slot", 0) or 0) for item in items],
    }

    try:
        raw = invoke_structured_with_slim_trace(
            _grouped_planner_llm,
            messages,
            invoke_config=invoke_config,
            trace_name=(invoke_config or {}).get("run_name", f"orchestrator.synthesis_plan.{op_type}_group.llm"),
            tags=list((invoke_config or {}).get("tags", []) or []),
            metadata=dict((invoke_config or {}).get("metadata", {}) or {}),
            summary_inputs=summary_inputs,
            output_key="grouped_synthesis_plan",
        )
        plan_map: Dict[int, Dict[str, Any]] = {
            int(p.get("slot", 0) or 0): p
            for p in (raw.get("slot_plans") or [])
        }
        output = []
        for item in items:
            slot = int(item.get("slot", 0) or 0)
            plan = dict(plan_map.get(slot) or fallbacks[slot])
            # Fill any empty fields with deterministic fallback values.
            for key, value in fallbacks[slot].items():
                if plan.get(key) in (None, "", []):
                    plan[key] = value
            if "research_required" not in plan or plan.get("research_required") is None:
                plan["research_required"] = bool((plan.get("research_focus") or "").strip())
            output.append(SynthesisPlanSchema.model_validate(plan).model_dump())
        return output
    except Exception as exc:
        logger.warning(f"Grouped synthesis plan LLM fallback for op_type={op_type}: {exc}")
        return [fallbacks[int(item.get("slot", 0) or 0)] for item in items]
