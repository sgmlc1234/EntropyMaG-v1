"""Deterministic synthesis-brief preparation.

Phase D2a (2026-04-16): the brief_validator LLM call and the separate
SynthesisBriefSchema have both been removed. A "brief" is now nothing more
than a SynthesisPlan dict with four research-overlay fields filled in:
``allowed_claims``, ``research_value``, ``research_usage_expectation``, and
``retry_focus``. Everything else comes straight from the pre-research
synthesis plan, so there is nothing left for an LLM to validate.

The fallback construction (``_fallback_brief``) already produced the full
dict deterministically; it is now the only path.
"""
import logging
import math
import re
from typing import Dict, List, Optional

from deepagent.memory_bank import derive_family_policy

logger = logging.getLogger(__name__)


def _split_claims(text: str) -> List[str]:
    return [part.strip(" -") for part in (text or "").replace("\n", " ").split(".") if part.strip()][:3]


def _research_value(artifact: Dict) -> str:
    """Tier the artifact on both source count AND content substance."""
    if not artifact or artifact.get("degraded"):
        return "low"
    if artifact.get("tool_used") == "arxiv_search" and int(artifact.get("evidence_block_count", 0) or 0) <= 0:
        return "low"
    source_count = len(artifact.get("sources", []) or [])
    idea_count = len([p for p in (artifact.get("idea_candidates") or []) if isinstance(p, str) and p.strip()])
    synth = (artifact.get("short_synthesis") or "").strip()
    synth_len = len(synth)
    has_conflict = bool((artifact.get("conflict_note") or "").strip())
    if idea_count >= 4 and synth_len >= 120 and not has_conflict:
        return "high"
    if idea_count >= 2 and synth_len >= 60:
        return "medium"
    if synth_len < 80:
        return "low"
    if source_count >= 3 and synth_len >= 200 and not has_conflict:
        return "high"
    if source_count >= 1 and synth_len >= 80:
        return "medium"
    return "low"


def _allowed_claims(artifact: Dict) -> List[str]:
    """Prefer idea_candidates (noun-phrase techniques) over period-split synthesis."""
    if (artifact or {}).get("tool_used") == "arxiv_search" and int((artifact or {}).get("evidence_block_count", 0) or 0) <= 0:
        return []
    idea_candidates = [
        str(phrase).strip(" -")
        for phrase in ((artifact or {}).get("idea_candidates") or [])
        if isinstance(phrase, str) and str(phrase).strip()
    ][:3]
    if idea_candidates:
        return idea_candidates
    claims = _split_claims((artifact or {}).get("short_synthesis", ""))
    if claims:
        return claims
    query = (artifact or {}).get("query")
    if query:
        return [f"Use only techniques relevant to: {query}"]
    return []


def _research_usage_expectation(artifact: Dict) -> str:
    if (artifact or {}).get("degraded"):
        return "Research is degraded. Use invariant-first reasoning."
    if (artifact or {}).get("tool_used") == "arxiv_search" and int((artifact or {}).get("evidence_block_count", 0) or 0) <= 0:
        return "arXiv returned no theorem/proof body evidence. Treat it as a weak idea hint only; parent invariants and verification code decide."
    return "Treat the synthesis plan as authoritative and use research only to support approved claims."


def _numeric_parent_answers(work_item: Dict) -> List[int]:
    values: List[int] = []
    for parent in work_item.get("parents", []) or []:
        answer = str(parent.get("answer", "")).strip()
        if re.fullmatch(r"-?\d+", answer):
            values.append(int(answer))
    return values


def _modular_divisor_bridge_warning(work_item: Dict, brief: Dict) -> str:
    """Catch the common impossible-bridge case without adding a full verifier.

    Example: forcing a ratio-derived answer x=5 to satisfy x ≡ 0 (mod n)
    where n is a nontrivial divisor of 284. Since gcd(5, 284)=1, the bridge
    is only true for n=1 and should be replanned or redirected to a derived
    intermediate value such as s(284).
    """
    text = " ".join(
        str(value or "")
        for value in [
            work_item.get("variation_axis", ""),
            brief.get("relation_guard", ""),
            brief.get("concept_to_activate", ""),
            brief.get("parameter_reuse_policy", ""),
            brief.get("deep_variant_requirement", ""),
        ]
    ).lower()
    if "divisor" not in text or ("mod" not in text and "congru" not in text):
        return ""
    divisor_targets = [
        int(match)
        for match in re.findall(r"divisors?\s+of\s+\$?(-?\d+)\$?", text)
        if match not in {"0", "1", "-1"}
    ]
    if not divisor_targets:
        return ""
    for answer in _numeric_parent_answers(work_item):
        if abs(answer) <= 1:
            continue
        for target in divisor_targets:
            if abs(target) <= 1:
                continue
            if math.gcd(abs(answer), abs(target)) == 1:
                return (
                    f"Bridge feasibility warning: numeric parent answer {answer} has no nontrivial common divisor with {target}. "
                    "Do not force a modular-divisor bridge on that target; use a verified derived intermediate value or emit bridge_missing."
                )
    return ""


def _fallback_brief(
    work_item: Dict,
    pair_plan: Dict,
    artifact: Dict,
    brief_role: str,
    context_pack: Optional[Dict] = None,
    retry_feedback: str = "",
) -> Dict:
    """Compute the research-independent guard fields deterministically.

    This is the baseline "brief" shape: plan-level guards + research overlays.
    When a synthesis_plan is supplied downstream in ``prepare_synthesis_brief``,
    its authoritative fields overlay this baseline.
    """
    context_pack = dict(context_pack or {})
    allowed_claims = _allowed_claims(artifact or {})
    opportunity = dict(context_pack.get("opportunity_context", {}) or {})
    remaining_note = (opportunity.get("remaining_concept_note", "") or "").strip()
    preferred_delta = (opportunity.get("preferred_delta", "") or "").strip()  # noqa: F841
    preferred_pattern = (opportunity.get("preferred_composition_pattern", "") or "").strip() or "serial_pipeline"
    underexplored_family_forms = list(opportunity.get("underexplored_family_forms", []) or [])[:2]
    parameter_reuse_policy = (opportunity.get("parameter_reuse_policy", "") or "").strip()
    deep_variant_requirement = (opportunity.get("deep_variant_requirement", "") or "").strip()

    target_quantity_guard = "; ".join(
        bundle.get("target_quantity", "")
        for bundle in (work_item.get("invariant_bundles", []) or pair_plan.get("invariant_bundles", []) or [])[:2]
        if bundle.get("target_quantity")
    ) or "Keep the parent target quantity explicit and unchanged."

    parent_statements = " ".join(
        str(parent.get("statement", "")) for parent in (work_item.get("parents", []) or [])
    ).lower()
    relation_guard = "Preserve the exact relation operators and quantifiers from the parent statements."
    if " exactly " in f" {parent_statements} " or " is equal to " in f" {parent_statements} " or " iff " in f" {parent_statements} ":
        relation_guard = (
            "Preserve exact equality and exact set-identity semantics from the parent. "
            "Do not weaken 'is equal to' / 'exactly' / iff into subset, containment, existence, or one-way implication claims."
        )
    if retry_feedback and "exactly" in retry_feedback.lower():
        relation_guard = (
            "Preserve the exact relation called out in retry feedback. "
            "Do not weaken an exact equality or exact set condition into a merely sufficient or subset-style statement."
        )

    concept_to_activate = remaining_note or (opportunity.get("underexplored_axes", ["Use an underexplored concept."]) or ["Use an underexplored concept."])[0]
    pattern_to_avoid = next(
        (
            ", ".join((contrast.get("failure_signatures") or [])[:2])
            for contrast in (context_pack.get("contrast_context", []) or [])
            if contrast.get("failure_signatures")
        ),
        "Avoid near-copy and repeated shallow variants.",
    )

    family_signatures = [
        card.get("family_signature", "")
        for card in (context_pack.get("lineage_context", []) or [])[:2]
        if card.get("family_signature")
    ]
    family_policy = derive_family_policy(family_signatures)
    parameter_reuse_policy = parameter_reuse_policy or family_policy["parameter_reuse_policy"]
    if not deep_variant_requirement:
        deep_variant_requirement = family_policy["deep_variant_requirement"]
        if deep_variant_requirement == "Prefer a deeper structural change over a shallow coefficient reshuffle.":
            deep_variant_requirement = f"Prefer {preferred_pattern} over weighted_sum_shuffle."
    if underexplored_family_forms:
        deep_variant_requirement = (
            deep_variant_requirement
            + f" Underexplored family/forms: {', '.join(underexplored_family_forms)}."
        )

    return {
        "brief_role": "mutation" if brief_role == "mutation" else "crossover",
        "pair_id": pair_plan.get("pair_id") or work_item.get("pair_id") or "",
        "slot": int(work_item.get("slot", pair_plan.get("mutation_slot", 0)) or 0),
        "research_value": _research_value(artifact or {}),
        "allowed_claims": allowed_claims,
        "target_quantity_guard": target_quantity_guard,
        "relation_guard": relation_guard,
        "concept_to_activate": concept_to_activate,
        "pattern_to_avoid": pattern_to_avoid,
        "preferred_composition_pattern": preferred_pattern,
        "parameter_reuse_policy": parameter_reuse_policy,
        "deep_variant_requirement": deep_variant_requirement,
        "research_usage_expectation": _research_usage_expectation(artifact or {}),
        "retry_focus": retry_feedback or "",
    }


def _apply_synthesis_plan_to_brief(brief: Dict, synthesis_plan: Dict) -> Dict:
    """Overlay the pre-research synthesis plan onto a brief.

    Plan is authoritative for the research-independent fields; research-dependent
    fields (allowed_claims, research_value, research_usage_expectation,
    retry_focus) are left to the brief overlay.
    """
    if not synthesis_plan:
        return brief
    plan_fields = (
        "target_quantity_guard",
        "relation_guard",
        "concept_to_activate",
        "pattern_to_avoid",
        "preferred_composition_pattern",
        "parameter_reuse_policy",
        "deep_variant_requirement",
    )
    overlaid = dict(brief)
    for key in plan_fields:
        value = synthesis_plan.get(key)
        if value:
            overlaid[key] = value
    return overlaid


def prepare_synthesis_brief(
    work_item: Dict,
    pair_plan: Dict,
    artifact: Dict,
    brief_role: str,
    context_pack: Optional[Dict] = None,
    retry_feedback: str = "",
    invoke_config: Optional[Dict] = None,
    synthesis_plan: Optional[Dict] = None,
) -> Dict:
    """Build the generator-facing brief deterministically.

    Phase D2a: the brief_validator LLM has been retired. Given an authoritative
    pre-research synthesis plan plus the research artifact, there is nothing
    left for an LLM to check — the plan is authoritative for plan-side fields,
    and the research overlay is a mechanical transformation of the artifact.
    Saves one LLM call per non-survivor slot per cycle.
    """
    # invoke_config is accepted for signature compatibility but ignored: no LLM.
    _ = invoke_config
    fallback = _fallback_brief(
        work_item,
        pair_plan,
        artifact,
        brief_role,
        context_pack=context_pack,
        retry_feedback=retry_feedback,
    )
    brief = _apply_synthesis_plan_to_brief(fallback, dict(synthesis_plan or {}))
    bridge_warning = _modular_divisor_bridge_warning(work_item, brief)
    if bridge_warning:
        brief["bridge_feasibility_note"] = bridge_warning
        brief["relation_guard"] = f"{brief.get('relation_guard', '').rstrip()} {bridge_warning}".strip()
        brief["research_usage_expectation"] = (
            f"{brief.get('research_usage_expectation', '').rstrip()} {bridge_warning}".strip()
        )
    return brief
