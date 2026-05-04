import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_paths import extract_generation_number, list_generation_files, load_problem_file, load_seed_problems
from deepagent.family_strategy import infer_family_signature
from deepagent.invariants import extract_invariant_bundle
from prompts import (
    ContextPackSchema,
    GenerationMemoryCardSchema,
    ProblemMemoryCardSchema,
    RunWorkingMemorySchema,
    ValidationEvidencePackSchema,
)

MEMORY_DIR = Path("data/memory")
REPO_ROOT = Path(__file__).resolve().parents[1]

STAGE_LIMITS = {
    "selector": {"max_cards": 4, "max_tokens": 400},
    "researcher": {"max_cards": 4, "max_tokens": 600},
    "briefing": {"max_cards": 4, "max_tokens": 500},
    "generator": {"lineage_cards": 1, "contrast_cards": 2, "max_tokens": 350},
}


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _compact(text: str, limit: int = 220) -> str:
    return " ".join((text or "").split())[:limit]


def _estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return max(1, len(text) // 4)


def _digest(payload: Any) -> str:
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _difficulty_text(problem: Dict[str, Any]) -> str:
    difficulty = problem.get("difficulty", "")
    return str(difficulty if difficulty is not None else "")


def _listify(values: Any) -> List[str]:
    if isinstance(values, list):
        return [str(value) for value in values if str(value).strip()]
    if values in (None, ""):
        return []
    return [str(values)]


def _research_quality(problem: Dict[str, Any]) -> str:
    artifact = dict(problem.get("research_artifact", {}) or {})
    if artifact.get("degraded"):
        return "low"
    source_count = len(artifact.get("sources", []) or [])
    if source_count >= 3:
        return "high"
    if source_count >= 1:
        return "medium"
    return "low"


def _OBSOLETE_signature_pipeline_DELETED(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Sprint 3 Phase C — full deletion of the signature pipeline.

    The previous implementation (~150 lines: derive_structure_signatures +
    _novelty_risk_flags + signature scoring in retrieve_archival_evidence) was
    a regex-based heuristic that fed an over-engineered novelty hard-block.
    Replaced by orchestrator-managed novelty judgment (LLM verdict on top-K
    retrieved cards) in `ground_and_rescore_node`. Function preserved as a
    one-line stub so any external caller fails LOUD and EARLY rather than
    silently with empty defaults.
    """
    raise NotImplementedError(
        "derive_structure_signatures was removed in Sprint 3 Phase C. "
        "Novelty is now judged by the grounding gate (see "
        "memory_bank.retrieve_topk_archive_by_text + GroundingReviewSchema.novelty_verdict)."
    )


def derive_family_policy(family_signatures: List[str]) -> Dict[str, str]:
    joined = " ".join(family_signatures)
    if "symmetric_power_sum_system" in joined:
        return {
            "parameter_reuse_policy": "Prefer new solvable systems over parent-constant reuse.",
            "deep_variant_requirement": "Construct a new verified system with natural-number roots instead of coefficient shuffling.",
        }
    if "binomial_moment" in joined:
        return {
            "parameter_reuse_policy": "Prefer changing n, p, moment order, or the exact derived target.",
            "deep_variant_requirement": "Do not use the moment only as a helper constant donor.",
        }
    if "nearest_prime" in joined:
        return {
            "parameter_reuse_policy": "Reuse constants only when the new exact filter remains explicit and verifiable.",
            "deep_variant_requirement": "Prefer exact structural filters over larger raw N.",
        }
    return {
        "parameter_reuse_policy": "Avoid direct parent-constant reuse unless the statement explicitly derives it.",
        "deep_variant_requirement": "Prefer a deeper structural change over a shallow coefficient reshuffle.",
    }


def _failure_signatures(problem: Dict[str, Any]) -> List[str]:
    signatures = []
    signatures.extend(_listify(problem.get("failure_signatures")))
    if problem.get("failure_signature"):
        signatures.append(str(problem.get("failure_signature")))
    metrics = dict(problem.get("lineage_metrics", {}) or {})
    signatures.extend(_listify(metrics.get("recent_retry_reasons")))
    quality = dict(problem.get("quality_assessment", {}) or {})
    signatures.extend(_listify(quality.get("issues")))
    deduped = []
    seen = set()
    for signature in signatures:
        normalized = _normalize(signature)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped[:6]



def _concept_summary(problem: Dict[str, Any], bundle: Dict[str, Any]) -> str:
    parts = [
        bundle.get("named_definition") or bundle.get("target_quantity") or "preserve parent invariant",
        "axes=" + ", ".join((bundle.get("allowed_variation_axes") or [])[:2]) if bundle.get("allowed_variation_axes") else "",
        "statement=" + _compact(problem.get("statement", ""), 120),
    ]
    return " | ".join(part for part in parts if part)


def _decomposition_notes(problem: Dict[str, Any], bundle: Dict[str, Any]) -> str:
    parent_count = len(problem.get("parent_ids", []) or [])
    used_axis = problem.get("variation_axis_used") or problem.get("variation_axis") or ""
    bridge_axis = problem.get("bridge_axis") or ""
    remaining_axes = [axis for axis in (bundle.get("allowed_variation_axes") or []) if axis != used_axis]
    if parent_count >= 2:
        note = "two-concept bridge"
        if bridge_axis:
            note += f"; active bridge={bridge_axis}"
        if used_axis:
            note += f"; explicit variation={used_axis}"
        if remaining_axes:
            note += f"; remaining concept={remaining_axes[0]}"
        return note
    if problem.get("op_type") == "mutation":
        note = "single-parent decomposition"
        if used_axis:
            note += f"; kept axis={used_axis}"
        if remaining_axes:
            note += f"; leftover axis={remaining_axes[0]}"
        return note
    if problem.get("type") == "survivor" or problem.get("op_type") == "survivor":
        return "survivor carry-over; preserve strongest invariant anchor"
    return "seed anchor for future decomposition"


def _ancestor_ids(problem: Dict[str, Any]) -> List[str]:
    parent_ids = _listify(problem.get("parent_ids"))
    lineage = dict(problem.get("lineage_metrics", {}) or {})
    parent_ids.extend(_listify(lineage.get("recent_parent_ids")))
    deduped = []
    seen = set()
    for problem_id in parent_ids:
        if problem_id and problem_id not in seen:
            seen.add(problem_id)
            deduped.append(problem_id)
    return deduped[:6]


def _problem_card(problem: Dict[str, Any], generation: int, source_kind: str, invariant_bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build an archival memory card for a problem.

    Phase D1 (2026-04-16): slimmed to 9 fields. Signature hashes, decomposition
    notes, failure signatures, research_quality, answer_excerpt, and
    variation_axes are all gone — they were write-only debt. The merged
    ``summary`` field absorbs concept_summary + axes into one compact string
    (<=200 chars) usable for Jaccard text retrieval in the grounding gate.
    """
    bundle = dict(invariant_bundle or extract_invariant_bundle(problem))
    concept = _concept_summary(problem, bundle) or ""
    axes = list(bundle.get("allowed_variation_axes") or [])[:4]
    summary = concept
    if axes:
        suffix = " | axes: " + ", ".join(str(a) for a in axes)
        summary = (concept + suffix)[:200]
    else:
        summary = concept[:200]
    card = ProblemMemoryCardSchema.model_validate(
        {
            "problem_id": problem.get("id", ""),
            "generation": int(generation),
            "source_kind": source_kind,
            "parent_ids": _listify(problem.get("parent_ids")),
            "ancestor_ids": _ancestor_ids(problem),
            "difficulty": _difficulty_text(problem),
            "op_type": str(problem.get("op_type") or problem.get("type") or "unknown"),
            "summary": summary,
            "statement_excerpt": _compact(problem.get("statement", ""), 180),
        }
    )
    payload = card.model_dump()
    payload["memory_meta"] = {
        "bridge_axis": problem.get("bridge_axis", ""),
        "variation_axis_used": problem.get("variation_axis_used") or problem.get("variation_axis") or "",
        "problem_type": problem.get("type", ""),
    }
    return payload


_NOVELTY_TOKEN_RE = re.compile(r"[^0-9a-zA-Z]+")
_NOVELTY_STOPWORDS = frozenset({
    "the","a","an","and","or","of","for","to","in","on","is","are","be",
    "with","that","this","these","those","as","at","by","let",
    "find","compute","determine","prove","show","given","suppose","all",
    "any","some","each","every","if","then","we","such","where","which","from",
    "consider","problem","solution","answer",
})


def _novelty_tokens(text: str, min_len: int = 3) -> set:
    """Content-token bag for cheap Jaccard similarity (Sprint 3 simplification).

    Mirrors the validator's _salient_tokens spirit but returns a set for O(1)
    intersection. Math-salient short tokens (mod, gcd, lcm, sum, ...) are
    preserved by min_len=3.
    """
    return {
        tok for tok in _NOVELTY_TOKEN_RE.split((text or "").lower())
        if tok and len(tok) >= min_len and not tok.isdigit() and tok not in _NOVELTY_STOPWORDS
    }


def retrieve_topk_archive_by_text(
    candidate: Dict[str, Any],
    archival_memory_handle: Optional[Dict[str, Any]],
    *,
    k: int = 3,
    min_overlap: float = 0.05,
) -> List[Dict[str, Any]]:
    """Cheap text-similarity retrieval of top-K archival cards (Sprint 3).

    Replaces the old 5-signature scoring. Pure Jaccard over content tokens of
    candidate.statement vs each card.statement_excerpt + summary.
    Returned cards are ranked by similarity descending; cards below
    `min_overlap` are filtered out so the LLM judge isn't spammed with
    unrelated entries.

    Excludes the candidate's own card and any card that shares its problem_id.
    """
    archival_memory_handle = dict(archival_memory_handle or {})
    cards = [_normalize_problem_memory_card(card) for card in (archival_memory_handle.get("problem_cards", []) or [])]
    if not cards:
        return []
    candidate_text = " ".join([
        str(candidate.get("statement", "") or ""),
        str(candidate.get("solution", "") or ""),
    ])
    cand_tokens = _novelty_tokens(candidate_text)
    if not cand_tokens:
        return []
    candidate_id = candidate.get("id", "")
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for card in cards:
        if card.get("problem_id") == candidate_id:
            continue
        card_text = " ".join([
            str(card.get("statement_excerpt", "") or ""),
            # Phase D1: summary replaces the old concept_summary field.
            str(card.get("summary", "") or ""),
        ])
        card_tokens = _novelty_tokens(card_text)
        if not card_tokens:
            continue
        intersection = len(cand_tokens & card_tokens)
        union = len(cand_tokens | card_tokens)
        jaccard = intersection / union if union else 0.0
        if jaccard >= min_overlap:
            scored.append((jaccard, card))
    scored.sort(key=lambda entry: (-entry[0], entry[1].get("generation", 0)))
    return [
        {
            **card,
            "_retrieval_similarity": round(sim, 3),
        }
        for sim, card in scored[:k]
    ]


def _bundle_signatures(bundle: Dict[str, Any]) -> Tuple[str, str]:
    invariant_signature = _digest(
        {
            "named_definition": bundle.get("named_definition", ""),
            "core_relations": bundle.get("core_relations", []),
            "domain_constraints": bundle.get("domain_constraints", []),
        }
    )
    target_signature = _digest(bundle.get("target_quantity", ""))
    return invariant_signature, target_signature


def _normalize_problem_memory_card(card: Dict[str, Any]) -> Dict[str, Any]:
    """Sprint 3 Phase C: pass-through. Old setdefault for signature fields was
    removed because those fields are no longer part of ProblemMemoryCardSchema.
    Preserved as a function so existing callers keep working without churn.
    """
    return dict(card)


def _generation_card(generation: int, cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Phase D1 (2026-04-16): archive only 3 fields.

    Previously this emitted diversity_summary / repeated_patterns /
    underexplored_axes / motifs / family_forms / dominant_invariant_signatures.
    None of those were read by downstream code (runtime opportunity_context is
    computed independently in build_run_working_memory). Archive only
    long-lived facts: population_size and frequent failure signatures.
    """
    failure_counter: Counter = Counter()
    for card in cards:
        # Aggregate failure signatures from per-card memory_meta where we still
        # record them; source cards no longer carry a failure_signatures field
        # directly.
        meta = card.get("memory_meta", {}) or {}
        for sig in meta.get("failure_signatures", []) or []:
            failure_counter[sig] += 1
    return GenerationMemoryCardSchema.model_validate(
        {
            "generation": int(generation),
            "population_size": len(cards),
            "frequent_failure_signatures": [value for value, _ in failure_counter.most_common(5)],
        }
    ).model_dump()


def persist_memory_bank(memory_bank: Dict[str, Any], base_dir: Path = MEMORY_DIR) -> Dict[str, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    bank_path = base_dir / "memory_bank.json"
    cards_path = base_dir / "problem_cards.jsonl"
    generations_path = base_dir / "generation_cards.json"
    plan_outcomes_path = base_dir / "plan_outcome_cards.json"
    bank_path.write_text(json.dumps(memory_bank, ensure_ascii=False, indent=2), encoding="utf-8")
    with cards_path.open("w", encoding="utf-8") as handle:
        for card in memory_bank.get("problem_cards", []):
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    generations_path.write_text(
        json.dumps(memory_bank.get("generation_cards", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plan_outcomes_path.write_text(
        json.dumps(memory_bank.get("plan_outcome_cards", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "memory_bank": str(bank_path),
        "problem_cards": str(cards_path),
        "generation_cards": str(generations_path),
        "plan_outcome_cards": str(plan_outcomes_path),
    }


def load_persisted_memory_bank(base_dir: Path = MEMORY_DIR) -> Dict[str, Any]:
    bank_path = base_dir / "memory_bank.json"
    if not bank_path.exists():
        return {}
    try:
        return json.loads(bank_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def should_persist_memory_bank(session_options: Dict[str, Any]) -> bool:
    artifact_dir = str((session_options or {}).get("artifact_dir", "") or "")
    run_name = Path(artifact_dir).name if artifact_dir else ""
    return not run_name.startswith("deep-trace-")


def _should_load_persisted_memory(session_options: Dict[str, Any]) -> bool:
    if not should_persist_memory_bank(session_options):
        return False
    artifact_dir = (session_options or {}).get("artifact_dir")
    save_format = (session_options or {}).get("save_format")
    candidate = artifact_dir or save_format
    if not candidate:
        return False
    try:
        candidate_path = Path(str(candidate)).expanduser().resolve()
        candidate_path.relative_to(REPO_ROOT)
        return True
    except Exception:
        return False


def _load_archival_from_history(session_options: Dict[str, Any]) -> Dict[str, Any]:
    cards_by_id: Dict[str, Dict[str, Any]] = {}

    def add_problem(problem: Dict[str, Any], generation: int, source_kind: str):
        problem_id = problem.get("id")
        if not problem_id:
            return
        cards_by_id[problem_id] = _problem_card(problem, generation=generation, source_kind=source_kind)

    seed_spec = session_options.get("seed_spec")
    if seed_spec:
        try:
            for seed_problem in load_seed_problems(seed_spec):
                add_problem(seed_problem, generation=0, source_kind="seed")
        except Exception:
            pass

    for generation_path in list_generation_files(save_format=session_options.get("save_format")):
        try:
            generation_number = max(extract_generation_number(generation_path), 0)
            for problem in load_problem_file(generation_path):
                add_problem(problem, generation=generation_number, source_kind="generated")
        except Exception:
            continue

    cards = list(cards_by_id.values())
    by_generation: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for card in cards:
        by_generation[int(card.get("generation", 0) or 0)].append(card)
    generation_cards = [_generation_card(gen, by_generation[gen]) for gen in sorted(by_generation)]
    return {
        "problem_cards": cards,
        "generation_cards": generation_cards,
        "metrics": {
            "problem_card_count": len(cards),
            "generation_card_count": len(generation_cards),
            "source": "reconstructed",
        },
        "paths": {},
    }


def load_archival_memory_handle(session_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    session_options = dict(session_options or {})
    persisted = load_persisted_memory_bank() if _should_load_persisted_memory(session_options) else {}
    if persisted.get("problem_cards") or persisted.get("generation_cards"):
        cards = [_normalize_problem_memory_card(card) for card in (persisted.get("problem_cards", []) or [])]
        handle = {
            "problem_cards": cards,
            "generation_cards": list(persisted.get("generation_cards", []) or []),
            "plan_outcome_cards": list(persisted.get("plan_outcome_cards", []) or []),
            "metrics": {
                "problem_card_count": len(cards),
                "generation_card_count": len(persisted.get("generation_cards", []) or []),
                "plan_outcome_card_count": len(persisted.get("plan_outcome_cards", []) or []),
                "source": "persisted",
            },
            "paths": dict(persisted.get("paths", {}) or {}),
        }
        return handle
    return _load_archival_from_history(session_options)


def _trim_cards(cards: List[Dict[str, Any]], max_cards: int, max_tokens: int) -> Tuple[List[Dict[str, Any]], int]:
    trimmed = []
    token_total = 0
    for card in cards[:max_cards]:
        estimate = _estimate_tokens(card)
        if trimmed and token_total + estimate > max_tokens:
            break
        trimmed.append(card)
        token_total += estimate
    return trimmed, token_total


def _build_run_stage_views(base_pack: Dict[str, Any]) -> Dict[str, Any]:
    """Phase D3a (2026-04-16): slim stage views.

    Previously each view re-stored authoritative_core / opportunity_context /
    research_policy from the base pack, duplicating ~4-5KB per slot. Those
    fields are now read straight from the base pack by build_context_pack_block
    (see prompts/__init__.py), so stage_views only has to carry what is
    UNIQUE per stage: the per-stage-trimmed card lists plus a token estimate.
    """
    lineage_cards = list(base_pack.get("lineage_context", []) or [])
    contrast_cards = list(base_pack.get("contrast_context", []) or [])
    researcher_cards, researcher_tokens = _trim_cards(
        lineage_cards + contrast_cards,
        STAGE_LIMITS["researcher"]["max_cards"],
        STAGE_LIMITS["researcher"]["max_tokens"],
    )
    briefing_cards, briefing_tokens = _trim_cards(
        lineage_cards + contrast_cards,
        STAGE_LIMITS["briefing"]["max_cards"],
        STAGE_LIMITS["briefing"]["max_tokens"],
    )
    generator_lineage, _ = _trim_cards(
        lineage_cards,
        STAGE_LIMITS["generator"]["lineage_cards"],
        STAGE_LIMITS["generator"]["max_tokens"],
    )
    generator_contrast, _ = _trim_cards(
        contrast_cards,
        STAGE_LIMITS["generator"]["contrast_cards"],
        STAGE_LIMITS["generator"]["max_tokens"],
    )
    selector_contrast = contrast_cards[:1]
    selector_view = {"contrast_context": selector_contrast}
    researcher_view = {"cards": researcher_cards}
    briefing_view = {"cards": briefing_cards}
    generator_view = {
        "lineage_context": generator_lineage,
        "contrast_context": generator_contrast,
    }
    selector_view["token_estimate"] = _estimate_tokens(selector_view)
    researcher_view["token_estimate"] = max(researcher_tokens, _estimate_tokens(researcher_view))
    briefing_view["token_estimate"] = max(briefing_tokens, _estimate_tokens(briefing_view))
    generator_view["token_estimate"] = _estimate_tokens(generator_view)
    return {
        "selector": selector_view,
        "researcher": researcher_view,
        "briefing": briefing_view,
        "generator": generator_view,
    }


def summarize_plan_outcome_cards(cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate plan_outcome_cards into a compact selector-facing summary."""
    if not cards:
        return {
            "recent_cards": [],
            "aggregate": {},
            "axes_planned_but_not_realized": [],
            "compositions_planned_but_not_realized": [],
            "recurrent_failure_signatures": [],
            "op_type_stats": {},
        }
    recent = sorted(cards, key=lambda card: int(card.get("generation", 0) or 0))[-12:]
    axes_not_realized: List[str] = []
    compositions_not_realized: List[str] = []
    failure_counter: Counter = Counter()
    status_counter: Counter = Counter()
    composition_success: Counter = Counter()
    composition_attempts: Counter = Counter()
    op_attempts: Counter = Counter()
    op_saved: Counter = Counter()
    op_axis_realized: Counter = Counter()
    op_composition_realized: Counter = Counter()
    op_repair_total: Counter = Counter()
    op_failed: Counter = Counter()
    for card in cards:
        status_counter[card.get("actual_status", "unknown")] += 1
        op_type = card.get("op_type", "")
        if op_type in {"mutation", "crossover"}:
            op_attempts[op_type] += 1
            status = card.get("actual_status", "")
            if status in {"saved", "repaired", "elite_backfill"}:
                op_saved[op_type] += 1
            elif status == "failed":
                op_failed[op_type] += 1
            if card.get("axis_realized", False):
                op_axis_realized[op_type] += 1
            if card.get("composition_realized", False):
                op_composition_realized[op_type] += 1
            try:
                op_repair_total[op_type] += int(card.get("repair_passes", 0) or 0)
            except Exception:
                pass
        if op_type != "survivor" and card.get("planned_variation_axis") and not card.get("axis_realized", False):
            axis = _compact(card.get("planned_variation_axis", ""), 140)
            if axis and axis not in axes_not_realized:
                axes_not_realized.append(axis)
        planned_composition = card.get("planned_composition_pattern", "") or ""
        if op_type != "survivor" and planned_composition:
            composition_attempts[planned_composition] += 1
            if card.get("composition_realized", False):
                composition_success[planned_composition] += 1
            else:
                if planned_composition not in compositions_not_realized:
                    compositions_not_realized.append(planned_composition)
        for signature in card.get("failure_signatures", []) or []:
            failure_counter[_compact(str(signature), 140)] += 1
    recurrent = [sig for sig, count in failure_counter.most_common(5) if count >= 2]
    composition_rates = {
        pattern: {
            "attempts": composition_attempts[pattern],
            "realized": composition_success[pattern],
        }
        for pattern in composition_attempts
    }
    op_type_stats: Dict[str, Dict[str, Any]] = {}
    for op_type in ("mutation", "crossover"):
        attempts = op_attempts[op_type]
        if attempts == 0:
            continue
        saved = op_saved[op_type]
        axis = op_axis_realized[op_type]
        comp = op_composition_realized[op_type]
        repair_total = op_repair_total[op_type]
        op_type_stats[op_type] = {
            "attempts": attempts,
            "saved": saved,
            "failed": op_failed[op_type],
            "axis_realized": axis,
            "composition_realized": comp,
            "avg_repair_passes": round(repair_total / attempts, 2) if attempts else 0.0,
            "saved_rate": round(saved / attempts, 3) if attempts else 0.0,
            "axis_realized_rate": round(axis / attempts, 3) if attempts else 0.0,
            "composition_realized_rate": round(comp / attempts, 3) if attempts else 0.0,
        }
    return {
        "recent_cards": recent,
        "aggregate": {
            "total_cards": len(cards),
            "status_counts": dict(status_counter),
            "composition_rates": composition_rates,
        },
        "axes_planned_but_not_realized": axes_not_realized[:6],
        "compositions_planned_but_not_realized": compositions_not_realized[:6],
        "recurrent_failure_signatures": recurrent,
        "op_type_stats": op_type_stats,
    }


OP_ALLOCATION_MIN_ATTEMPTS = 3
OP_ALLOCATION_SCORE_FLOOR = 0.15  # keep at least this much weight on each op_type when both have data


def recommend_op_type_allocation(
    plan_outcome_cards: List[Dict[str, Any]],
    *,
    seed_count: int,
    non_survivor_slots: int,
) -> Dict[str, Any]:
    """Recommend mutation/crossover split for the next generation.

    The recommendation is soft: it surfaces as a hint in the selector prompt
    alongside the supporting stats. The selector LLM remains free to override
    with a documented rationale (enforced structurally by parent-count rules,
    but no quota is imposed by the repair gate).

    Decision order:
    1. If crossover is infeasible (seed_count < 2), allocate all slots to mutation.
    2. If non_survivor_slots <= 0, return zeros.
    3. If either op_type has >= OP_ALLOCATION_MIN_ATTEMPTS of history, weight by
       observed saved_rate with a small floor so we never zero out an op_type.
    4. Otherwise fall back to a balanced default (slight mutation bias).
    """
    result: Dict[str, Any] = {
        "mutation": 0,
        "crossover": 0,
        "rationale": "",
        "confidence": "low",
        "stats_available": False,
        "observed_attempts": {"mutation": 0, "crossover": 0},
    }
    if non_survivor_slots <= 0:
        result["rationale"] = "No non-survivor slots to allocate."
        return result
    if seed_count < 2:
        result["mutation"] = non_survivor_slots
        result["crossover"] = 0
        result["rationale"] = (
            f"Only {seed_count} seed available; crossover needs two distinct parents, "
            "so every non-survivor slot must be a mutation."
        )
        result["confidence"] = "hard_constraint"
        return result

    summary = summarize_plan_outcome_cards(list(plan_outcome_cards or []))
    stats = summary.get("op_type_stats", {}) or {}
    mutation_stats = stats.get("mutation", {}) or {}
    crossover_stats = stats.get("crossover", {}) or {}
    mutation_attempts = int(mutation_stats.get("attempts", 0) or 0)
    crossover_attempts = int(crossover_stats.get("attempts", 0) or 0)
    result["observed_attempts"] = {
        "mutation": mutation_attempts,
        "crossover": crossover_attempts,
    }

    both_have_data = (
        mutation_attempts >= OP_ALLOCATION_MIN_ATTEMPTS
        and crossover_attempts >= OP_ALLOCATION_MIN_ATTEMPTS
    )
    if both_have_data:
        result["stats_available"] = True
        mutation_score = float(mutation_stats.get("saved_rate", 0.0) or 0.0)
        crossover_score = float(crossover_stats.get("saved_rate", 0.0) or 0.0)
        # Keep both op_types in the mix: never zero-weight a viable operation.
        mutation_score = max(mutation_score, OP_ALLOCATION_SCORE_FLOOR)
        crossover_score = max(crossover_score, OP_ALLOCATION_SCORE_FLOOR)
        total_score = mutation_score + crossover_score
        mutation_share = mutation_score / total_score if total_score else 0.5
        raw_mutation = mutation_share * non_survivor_slots
        mutation_slots = max(1, min(non_survivor_slots - 1, int(round(raw_mutation))))
        crossover_slots = non_survivor_slots - mutation_slots
        result["mutation"] = mutation_slots
        result["crossover"] = crossover_slots
        result["confidence"] = "data_driven"
        result["rationale"] = (
            f"Historical saved_rate — mutation {mutation_stats.get('saved_rate', 0)} ({mutation_attempts} attempts), "
            f"crossover {crossover_stats.get('saved_rate', 0)} ({crossover_attempts} attempts). "
            f"Weighted split with floor {OP_ALLOCATION_SCORE_FLOOR} suggests {mutation_slots}/{crossover_slots}."
        )
        return result

    # Balanced default with slight mutation bias: half + remainder to mutation.
    mutation_slots = non_survivor_slots // 2 + non_survivor_slots % 2
    crossover_slots = non_survivor_slots - mutation_slots
    result["mutation"] = mutation_slots
    result["crossover"] = crossover_slots
    result["rationale"] = (
        f"Insufficient plan_outcome history (mutation={mutation_attempts}, "
        f"crossover={crossover_attempts}; need >= {OP_ALLOCATION_MIN_ATTEMPTS} each). "
        f"Fall back to balanced default with mutation bias on the odd slot."
    )
    result["confidence"] = "default"
    return result


def build_run_working_memory(
    work_items: List[Dict[str, Any]],
    current_generation: List[Dict[str, Any]],
    generation_count: int,
    invariant_bundles: Optional[Dict[str, Dict[str, Any]]] = None,
    selection_strategy: Optional[Dict[str, Any]] = None,
    failed_problems: Optional[List[Dict[str, Any]]] = None,
    slot_retry_registry: Optional[Dict[str, Any]] = None,
    accepted_candidate_deltas: Optional[List[Dict[str, Any]]] = None,
    plan_outcome_cards: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    invariant_bundles = dict(invariant_bundles or {})
    failed_problems = list(failed_problems or [])
    slot_retry_registry = dict(slot_retry_registry or {})
    current_parent_map = {problem.get("id"): problem for problem in current_generation or []}
    context_packs = {}
    metrics = {"context_token_size": {}, "pack_digests": {}}
    slot_notes: Dict[str, Dict[str, Any]] = {}

    for item in work_items or []:
        slot = int(item.get("slot", 0) or 0)
        parent_ids = _listify(item.get("parent_ids"))
        parents = [current_parent_map[parent_id] for parent_id in parent_ids if parent_id in current_parent_map]
        retry_signatures = list((slot_retry_registry.get(str(slot), {}) or {}).get("reason_signatures", []) or [])
        contrast_entries = [
            {
                "problem_id": failed.get("id", ""),
                "failure_signatures": _listify(failed.get("failure_signature") or failed.get("failure_signatures")),
                "statement_excerpt": _compact(failed.get("statement", ""), 180),
                "reason": failed.get("validation_feedback") or failed.get("constraint_guard_feedback") or "",
            }
            for failed in failed_problems
            if int(failed.get("_slot", failed.get("slot", -1)) or -1) == slot
        ][:2]
        unused_axes = []
        for bundle in item.get("invariant_bundles", []) or []:
            for axis in bundle.get("allowed_variation_axes", []) or []:
                if axis != item.get("variation_axis") and axis not in unused_axes:
                    unused_axes.append(axis)
        parent_families = [infer_family_signature(parent) for parent in parents if parent]
        family_strategy = derive_family_policy(parent_families)
        preferred_composition_pattern = (
            "same_system_new_parameters"
            if len(set(parent_families)) <= 1 and "symmetric_power_sum_system" in set(parent_families)
            else ("single_family_mutation" if len(set(parent_families)) <= 1 else "serial_pipeline")
        )
        authoritative_core = {
            "parent_summaries": [
                {
                    "id": parent.get("id", ""),
                    "statement_excerpt": _compact(parent.get("statement", ""), 220),
                    "answer": str(parent.get("answer", "")),
                }
                for parent in parents
            ],
            "invariant_bundles": item.get("invariant_bundles", []) or [],
            "validation_feedback": item.get("validation_feedback", ""),
            "constraint_guard_feedback": item.get("constraint_guard_feedback", ""),
        }
        opportunity_context = {
            "unused_axes": unused_axes[:3],
            "preferred_composition_pattern": preferred_composition_pattern,
            "parameter_reuse_policy": family_strategy["parameter_reuse_policy"],
            "deep_variant_requirement": family_strategy["deep_variant_requirement"],
            "remaining_concept_note": "",
            "preferred_delta": f"Preserve {item.get('variation_axis') or 'the assigned axis'} while using a fresh local transformation inside this run.",
        }
        forbidden_directions = []
        for bundle in item.get("invariant_bundles", []) or []:
            forbidden_directions.extend((bundle.get("forbidden_rewrites") or [])[:3])
        for contrast in contrast_entries:
            if contrast.get("failure_signatures"):
                forbidden_directions.append("avoid signatures: " + ", ".join(contrast.get("failure_signatures", [])[:2]))
        research_policy = {
            "requires_research": bool(item.get("requires_research", False)),
            "query_hint": item.get("query_hint")
            or f"Preserve parent invariants for {item.get('op_type', 'unknown')} using only current-run context.",
            "forbidden_directions": forbidden_directions[:5],
            "preferred_sources": ["tavily_search", "arxiv_search"],
        }
        base_pack = {
            "slot": slot,
            "op_type": item.get("op_type", "unknown"),
            "parent_ids": parent_ids,
            "authoritative_core": authoritative_core,
            "lineage_context": [
                _problem_card(
                    parent,
                    generation=generation_count,
                    source_kind="current",
                    invariant_bundle=invariant_bundles.get(parent.get("id", "")),
                )
                for parent in parents[:2]
            ],
            "contrast_context": contrast_entries[:2],
            "opportunity_context": opportunity_context,
            "research_policy": research_policy,
        }
        base_pack["stage_views"] = _build_run_stage_views(base_pack)
        base_pack["metrics"] = {
            "context_token_size": {
                stage: int(((view or {}).get("token_estimate")) or _estimate_tokens(view))
                for stage, view in (base_pack.get("stage_views", {}) or {}).items()
            },
            "retry_signature_count": len(retry_signatures),
        }
        base_pack["digest"] = _digest(
            {
                "slot": slot,
                "parent_ids": parent_ids,
                "retry_signatures": retry_signatures,
                "preferred_delta": opportunity_context.get("preferred_delta", ""),
            }
        )
        pack = ContextPackSchema.model_validate(base_pack).model_dump()
        context_packs[str(slot)] = pack
        metrics["context_token_size"][str(slot)] = pack["metrics"]["context_token_size"]
        metrics["pack_digests"][str(slot)] = pack["digest"]
        slot_notes[str(slot)] = {
            "failure_count": len(contrast_entries),
            "retry_signatures": retry_signatures[-4:],
            "latest_feedback": next((entry.get("reason", "") for entry in contrast_entries if entry.get("reason")), ""),
        }

    working_memory = RunWorkingMemorySchema.model_validate(
        {
            "run_goal": "Generate the next validated problem generation using only shared run-local memory.",
            "parent_invariant_cache": invariant_bundles,
            "generation_plan_summary": {
                "generation_count": generation_count,
                "selection_strategy": selection_strategy or {},
                "work_item_count": len(work_items or []),
            },
            "slot_notes": slot_notes,
            "retry_signatures": {
                key: list((value or {}).get("reason_signatures", []) or [])
                for key, value in slot_retry_registry.items()
            },
            "accepted_candidate_deltas": list(accepted_candidate_deltas or []),
            "run_summary": f"generation={generation_count}; work_items={len(work_items or [])}; current_generation={len(current_generation or [])}",
            "token_budget_meta": {
                stage: limits["max_tokens"] for stage, limits in STAGE_LIMITS.items()
            },
            "context_packs": context_packs,
            "stage_views": {
                "selector": {
                    "current_generation_size": len(current_generation or []),
                    "active_parent_ids": [problem.get("id", "") for problem in current_generation[:5]],
                    "slot_notes": slot_notes,
                    "selection_strategy": selection_strategy or {},
                    "plan_outcome_summary": summarize_plan_outcome_cards(list(plan_outcome_cards or [])),
                }
            },
            "metrics": metrics,
        }
    ).model_dump()
    return working_memory


def retrieve_archival_evidence(
    problem: Dict[str, Any],
    archival_memory_handle: Optional[Dict[str, Any]],
    generation_count: int,
) -> Dict[str, Any]:
    """Build the validator's archival_evidence block.

    Sprint 3 Phase B: simplified from 100-line signature scoring to plain
    text-similarity retrieval (Jaccard) reusing `retrieve_topk_archive_by_text`.
    `novelty_flags` is always [] — novelty is judged downstream by the
    grounding gate LLM, not here. This function only retrieves up to 3
    structurally similar archival cards as context for the validator's
    LLM-based regenerability check.
    """
    archival_memory_handle = dict(archival_memory_handle or {})
    cards = [_normalize_problem_memory_card(card) for card in (archival_memory_handle.get("problem_cards", []) or [])]
    if not cards:
        return ValidationEvidencePackSchema.model_validate(
            {
                "matched_problem_cards": [],
                "matched_generation_patterns": [],
                "novelty_flags": [],
                "similarity_rationale": "No archival evidence was available for validation.",
                "evidence_digest": _digest({"problem_id": problem.get("id", ""), "empty": True}),
            }
        ).model_dump()

    raw_matches = retrieve_topk_archive_by_text(problem, archival_memory_handle, k=3)
    # Strip the runtime-only _retrieval_similarity helper field before schema
    # validation so ValidationEvidencePackSchema (strict over card fields)
    # doesn't reject. The numeric similarity is folded into the rationale text
    # for human/LLM reading.
    matched_problem_cards = [
        {k: v for k, v in card.items() if not k.startswith("_")}
        for card in raw_matches
    ]
    rationale = (
        "Matched archival cards (text similarity): "
        + ", ".join(
            f"{c.get('problem_id','')}({c.get('_retrieval_similarity','?')})"
            for c in raw_matches[:3]
        )
        if raw_matches
        else "No textually similar archival entries were retrieved."
    )
    payload = {
        "matched_problem_cards": matched_problem_cards,
        "matched_generation_patterns": [],
        "novelty_flags": [],  # Sprint 3 Phase B: deprecated, always empty
        "similarity_rationale": rationale,
        "evidence_digest": _digest(
            {
                "problem_id": problem.get("id", ""),
                "matched_problem_ids": [c.get("problem_id", "") for c in matched_problem_cards],
            }
        ),
    }
    return ValidationEvidencePackSchema.model_validate(payload).model_dump()


def _condense_cards(cards_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    condensed = []
    for problem_id in sorted(cards_by_id):
        card = dict(cards_by_id[problem_id])
        # Phase D1: failure_signatures / novelty_risk_flags are no longer in the
        # card schema; drop any legacy keys if they leaked in from older code.
        card.pop("failure_signatures", None)
        card.pop("novelty_risk_flags", None)
        condensed.append(card)
    return condensed


PLAN_OUTCOME_WINDOW = 3  # retain only the most recent N generations of plan_outcome_cards


def _axis_realized(planned_axis: str, actual_axis: str) -> bool:
    """Loose token-overlap match between planned and realized variation axis."""
    planned = _normalize(planned_axis or "")
    actual = _normalize(actual_axis or "")
    if not planned:
        return False
    if not actual:
        return False
    if planned == actual:
        return True
    planned_tokens = {tok for tok in planned.split() if len(tok) >= 4}
    actual_tokens = {tok for tok in actual.split() if len(tok) >= 4}
    if not planned_tokens:
        return False
    overlap = planned_tokens & actual_tokens
    return len(overlap) >= max(1, min(2, len(planned_tokens) // 2))


def _composition_realized(planned_pattern: str, actual_brief_pattern: str) -> bool:
    """Exact-match check for the pre-research synthesis plan's composition pattern."""
    if not planned_pattern or not actual_brief_pattern:
        return False
    return _normalize(planned_pattern) == _normalize(actual_brief_pattern)


def _plan_outcome_cards(
    selection_strategy: Optional[Dict[str, Any]],
    work_items: Optional[List[Dict[str, Any]]],
    saved_problems: Optional[List[Dict[str, Any]]],
    failed_problems: Optional[List[Dict[str, Any]]],
    generation_count: int,
    saved_slot_map: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Build slot-level plan-vs-actual outcome cards for this generation.

    Each card records what the orchestrator planned for the slot and what
    actually happened, so the next generation's selector can read planned-but-
    not-realized axes and recurrent failure signatures.

    ``saved_slot_map`` (slot → problem_id) is used to recover slot assignments
    when ``save_generation`` strips ``_slot`` from problem dicts before disk
    persistence; without it, plan-outcome cards would degenerate to all-failed.
    """
    if not work_items:
        return []
    # Build slot → problem with three fallbacks in priority order:
    #   1. problem carries _slot (in-memory path before save strips it)
    #   2. problem carries slot (rare; e.g. repair path)
    #   3. saved_slot_map provides slot → id reverse lookup (post-save path)
    id_to_slot: Dict[str, int] = {}
    for slot, problem_id in (saved_slot_map or {}).items():
        try:
            if problem_id:
                id_to_slot[str(problem_id)] = int(slot)
        except Exception:
            continue
    saved_by_slot: Dict[int, Dict[str, Any]] = {}
    for problem in saved_problems or []:
        slot = problem.get("_slot", problem.get("slot"))
        if slot is None and problem.get("id") in id_to_slot:
            slot = id_to_slot[problem.get("id")]
        if slot is None:
            continue
        try:
            saved_by_slot[int(slot)] = problem
        except Exception:
            continue
    failed_by_slot: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for problem in failed_problems or []:
        slot = problem.get("_slot", problem.get("slot"))
        if slot is None:
            continue
        try:
            failed_by_slot[int(slot)].append(problem)
        except Exception:
            continue
    strategy_source = (selection_strategy or {}).get("strategy_source", "")
    cards: List[Dict[str, Any]] = []
    for item in work_items:
        try:
            slot = int(item.get("slot", 0) or 0)
        except Exception:
            slot = 0
        op_type = item.get("op_type", "")
        planned_axis = item.get("variation_axis", "") or ""
        planned_seed_focus = item.get("seed_focus", "") or ""
        planned_query_hint = item.get("query_hint", "") or ""
        synthesis_plan = dict(item.get("synthesis_plan", {}) or {})
        planned_composition = synthesis_plan.get("preferred_composition_pattern", "") or ""
        planned_parameter_reuse = synthesis_plan.get("parameter_reuse_policy", "") or ""
        planned_research_focus = synthesis_plan.get("research_focus", "") or ""
        saved = saved_by_slot.get(slot)
        failures = failed_by_slot.get(slot, [])
        if saved is not None:
            source_kind = saved.get("source_kind") or saved.get("_source_kind") or ""
            if source_kind == "elite_backfill":
                actual_status = "elite_backfill"
            elif failures:
                actual_status = "repaired"
            else:
                actual_status = "saved"
        elif op_type == "survivor":
            # Survivors may not appear in saved_problems by slot if they are
            # carried through untouched; treat presence of a carry as saved.
            actual_status = "saved"
        else:
            actual_status = "failed"
        actual_axis = ""
        if saved is not None:
            actual_axis = (
                saved.get("variation_axis_used")
                or saved.get("variation_axis")
                or ""
            )
        realized = bool(saved) and _axis_realized(planned_axis, actual_axis)
        actual_composition = ""
        if saved is not None:
            brief = (saved.get("synthesis_brief") or item.get("synthesis_brief") or {})
            actual_composition = brief.get("preferred_composition_pattern", "") or ""
        composition_realized = bool(saved) and _composition_realized(planned_composition, actual_composition)
        failure_signatures: List[str] = []
        for failure in failures:
            signature = failure.get("failure_signature") or failure.get("failure_mode")
            if signature:
                failure_signatures.append(_compact(str(signature), 140))
        cards.append(
            {
                "generation": int(generation_count or 0),
                "slot": slot,
                "pair_id": item.get("pair_id", ""),
                "op_type": op_type,
                "parent_ids": list(item.get("parent_ids", []) or []),
                "planned_variation_axis": planned_axis,
                "planned_seed_focus": planned_seed_focus,
                "planned_query_hint": planned_query_hint,
                "planned_mode": item.get("mode", ""),
                "planned_composition_pattern": planned_composition,
                "planned_parameter_reuse_policy": planned_parameter_reuse,
                "planned_research_focus": planned_research_focus,
                "actual_status": actual_status,
                "actual_variation_axis": actual_axis,
                "actual_composition_pattern": actual_composition,
                "axis_realized": realized,
                "composition_realized": composition_realized,
                "repair_passes": len(failures),
                "failure_signatures": list(dict.fromkeys(failure_signatures))[:3],
                "strategy_source": strategy_source,
            }
        )
    return cards


def _merge_plan_outcome_history(
    existing: Optional[List[Dict[str, Any]]],
    new_cards: List[Dict[str, Any]],
    *,
    window: int = PLAN_OUTCOME_WINDOW,
) -> List[Dict[str, Any]]:
    """Append this generation's plan_outcome_cards and keep only the latest N generations."""
    prior = list(existing or [])
    new_generations = {int(card.get("generation", 0) or 0) for card in new_cards}
    prior = [card for card in prior if int(card.get("generation", 0) or 0) not in new_generations]
    combined = prior + list(new_cards)
    if window and window > 0:
        by_gen: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for card in combined:
            by_gen[int(card.get("generation", 0) or 0)].append(card)
        kept_gens = sorted(by_gen.keys())[-window:]
        combined = [card for gen in kept_gens for card in by_gen[gen]]
    return combined


def consolidate_run_to_archive(
    current_generation: List[Dict[str, Any]],
    generation_count: int,
    session_options: Optional[Dict[str, Any]] = None,
    invariant_bundles: Optional[Dict[str, Dict[str, Any]]] = None,
    failed_problems: Optional[List[Dict[str, Any]]] = None,
    archival_memory_handle: Optional[Dict[str, Any]] = None,
    selection_strategy: Optional[Dict[str, Any]] = None,
    work_items: Optional[List[Dict[str, Any]]] = None,
    saved_slot_map: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    session_options = dict(session_options or {})
    invariant_bundles = dict(invariant_bundles or {})
    cards_by_id: Dict[str, Dict[str, Any]] = {}
    archival_memory_handle = archival_memory_handle or load_archival_memory_handle(session_options)
    for card in (archival_memory_handle.get("problem_cards", []) or []):
        normalized = _normalize_problem_memory_card(card)
        if normalized.get("problem_id"):
            cards_by_id[normalized["problem_id"]] = normalized

    def add_problem(problem: Dict[str, Any], source_kind: str):
        problem_id = problem.get("id")
        if not problem_id:
            return
        bundle = invariant_bundles.get(problem_id)
        cards_by_id[problem_id] = _problem_card(
            problem,
            generation=generation_count,
            source_kind=source_kind,
            invariant_bundle=bundle,
        )

    for problem in current_generation or []:
        add_problem(problem, "generated")
    for failed in failed_problems or []:
        add_problem(failed, "failed")

    cards = _condense_cards(cards_by_id)
    by_generation: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for card in cards:
        by_generation[int(card.get("generation", 0) or 0)].append(card)
    generation_cards = [_generation_card(gen, by_generation[gen]) for gen in sorted(by_generation)]
    new_outcome_cards = _plan_outcome_cards(
        selection_strategy=selection_strategy,
        work_items=work_items,
        saved_problems=current_generation,
        failed_problems=failed_problems,
        generation_count=generation_count,
        saved_slot_map=saved_slot_map,
    )
    prior_outcome_cards = archival_memory_handle.get("plan_outcome_cards", []) or []
    plan_outcome_cards = _merge_plan_outcome_history(prior_outcome_cards, new_outcome_cards)
    archive = {
        "problem_cards": cards,
        "generation_cards": generation_cards,
        "plan_outcome_cards": plan_outcome_cards,
        "metrics": {
            "problem_card_count": len(cards),
            "generation_card_count": len(generation_cards),
            "plan_outcome_card_count": len(plan_outcome_cards),
            "current_generation": generation_count,
        },
    }
    paths = persist_memory_bank(archive) if should_persist_memory_bank(session_options) else {}
    archive["paths"] = paths
    return archive


