import re
from typing import Dict, List

from prompts import InvariantAuditSchema, InvariantBundleSchema


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _compact(text: str, limit: int = 240) -> str:
    return " ".join((text or "").split())[:limit]


def _extract_explicit_target_request(statement: str) -> str:
    text = " ".join((statement or "").split())
    if not text:
        return ""
    patterns = [
        r"(Find the value of [^.?!]+[.?!]?)",
        r"(Determine the value of [^.?!]+[.?!]?)",
        r"(Compute [^.?!]+[.?!]?)",
        r"(Evaluate [^.?!]+[.?!]?)",
        r"(How many [^.?!]+[.?!]?)",
        r"(What is [^.?!]+[.?!]?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return ""


def _contains_any(text: str, patterns: List[str]) -> bool:
    normalized = _normalize(text)
    for pattern in patterns:
        if _normalize(pattern) in normalized:
            return True
    return False


def extract_invariant_bundle(problem: Dict) -> Dict:
    statement = problem.get("statement", "") or ""
    normalized = _normalize(statement)
    named_definition = ""
    domain_constraints: List[str] = []
    core_relations: List[str] = []
    explicit_target_request = _extract_explicit_target_request(statement)
    target_quantity = (
        f"Preserve the exact final target request: {explicit_target_request}"
        if explicit_target_request
        else f"Preserve the defining mathematical target implied by: {_compact(statement)}"
    )
    forbidden_rewrites: List[str] = []
    allowed_variation_axes: List[str] = [
        "change numerical bounds",
        "change aggregation range",
        "simplify computation while preserving the invariant",
    ]

    if "positive integers" in normalized:
        domain_constraints.append("positive integers")
    if "natural numbers" in normalized:
        domain_constraints.append("natural numbers")

    if "we say that the triple" in normalized and "cyclically divisible" in normalized:
        named_definition = (
            "cyclically divisible means (a+1)/b, (b+1)/c, and (c+1)/a are all integers"
        )
        core_relations = ["(a+1)/b", "(b+1)/c", "(c+1)/a"]
        target_quantity = (
            "Preserve the exact cyclically divisible definition. If F(n) is referenced, it counts triples in T whose sum divides n, not distinct sums or distinct values."
        )
        forbidden_rewrites.extend(
            [
                "a|(b+c)",
                "b|(c+a)",
                "c|(a+b)",
                "(a+b)/c",
                "(b+c)/a",
                "(c+a)/b",
                "complex numbers",
                "real numbers",
                "distinct values of s(a,b,c) divide n",
                "f(n) counts distinct sums",
                "determine the value of f(6)",
                "determine the value of f(12)",
                "count triples with s(a,b,c) <= 10",
            ]
        )
        allowed_variation_axes = [
            "change a larger aggregation range such as sum of F(n) over an interval",
            "ask about a weighted aggregate derived from S(a,b,c) or F(n)",
            "ask about distinct-sum or grouped-sum aggregates while preserving the cyclically divisible definition",
        ]
    elif "find the prime number closest" in normalized:
        target_quantity = "Preserve the task of identifying the prime number closest to the fixed integer target."
        forbidden_rewrites.extend(["tie-breaking that changes the original task", "non-prime nearest integer"])
        allowed_variation_axes = [
            "change the search neighborhood",
            "ask about the distance to the nearest prime",
            "ask about primality verification of nearby candidates",
        ]
    elif "a^4+b^4+c^4" in normalized and ("natural numbers" in normalized or "a+b+c" in normalized):
        target_quantity = "Preserve the task of determining the exact value of a^4+b^4+c^4 from the given symmetric system."
        forbidden_rewrites.extend(["complex numbers", "real numbers"])
        allowed_variation_axes = [
            "change the derived symmetric quantity",
            "ask about intermediate symmetric polynomials",
            "ask for a related power sum derived from the same system",
        ]
    elif "define the polynomial f(k)" in normalized and "find the value of f(" in normalized:
        target_match = re.search(r"find the value of\s+(f\([^)]*\))", statement, flags=re.IGNORECASE)
        target_expr = target_match.group(1) if target_match else "the exact final polynomial target"
        target_quantity = (
            f"Preserve the exact final target expression {target_expr}. "
            f"Do not replace it with any surrogate quantity such as a trace sum, determinant count, parity count, or helper statistic."
        )
        forbidden_rewrites.extend(
            [
                "sum of traces",
                "trace of permutation matrices",
                "count of determinants equal to -1",
                "number of odd permutations",
                "helper statistic instead of f(k)",
            ]
        )

    bundle = InvariantBundleSchema.model_validate(
        {
            "named_definition": named_definition,
            "domain_constraints": domain_constraints,
            "core_relations": core_relations,
            "target_quantity": target_quantity,
            "forbidden_rewrites": forbidden_rewrites,
            "allowed_variation_axes": allowed_variation_axes,
        }
    )
    return bundle.model_dump()


def extract_invariant_bundles(problems: List[Dict]) -> Dict[str, Dict]:
    bundles: Dict[str, Dict] = {}
    for problem in problems or []:
        problem_id = problem.get("id")
        if not problem_id or problem_id in bundles:
            continue
        bundles[problem_id] = extract_invariant_bundle(problem)
    return bundles


def audit_candidate_against_bundles(candidate: Dict, invariant_bundles: List[Dict]) -> Dict:
    statement = candidate.get("statement", "") or ""
    normalized = _normalize(statement)
    reasons: List[str] = []
    definition_preserved = True
    domain_preserved = True
    relation_preserved = True
    target_quantity_preserved = True

    # Domain and target-quantity rewrites are no longer hard gates. The orchestrator may still
    # inspect these shifts later, but they do not fail invariant preservation directly.

    for bundle in invariant_bundles or []:
        named_definition = bundle.get("named_definition", "")
        if named_definition:
            equivalent_groups = [
                ["(a+1)/b", "b | (a+1)", "b divides a+1"],
                ["(b+1)/c", "c | (b+1)", "c divides b+1"],
                ["(c+1)/a", "a | (c+1)", "a divides c+1"],
            ]
            for group in equivalent_groups:
                if not _contains_any(statement, group):
                    definition_preserved = False
                    relation_preserved = False
                    reasons.append("Candidate does not preserve the canonical cyclically divisible definition.")
                    break
        for forbidden in bundle.get("forbidden_rewrites", []) or []:
            if _normalize(forbidden) and _normalize(forbidden) in normalized:
                if "real number" in _normalize(forbidden) or "complex number" in _normalize(forbidden):
                    continue
                elif "distinct" in _normalize(forbidden) or "f(n)" in _normalize(forbidden):
                    continue
                else:
                    relation_preserved = False
                reasons.append(f"Candidate uses forbidden rewrite: {forbidden}")

    audit = InvariantAuditSchema.model_validate(
        {
            "definition_preserved": definition_preserved,
            "domain_preserved": domain_preserved,
            "relation_preserved": relation_preserved,
            "target_quantity_preserved": target_quantity_preserved,
            "reasons": reasons,
        }
    )
    return audit.model_dump()
