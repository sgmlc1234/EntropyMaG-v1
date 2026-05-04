import re
from typing import Any, Dict


FAMILY_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "permutation_matrix_trace": {
        "requires_research_default": False,
        "requires_briefing_default": False,
        "requires_solvability_gate": False,
        "preferred_composition_pattern": "single_family_mutation",
        "parameter_reuse_policy": "Preserve the exact permutation action and target trace semantics when changing constants.",
        "deep_variant_requirement": "Prefer new trace or iterate targets over superficial modulus or coefficient changes.",
        "preferred_repair_mode": "easier_rescue",
    },
    "code_automorphism": {
        "requires_research_default": False,
        "requires_briefing_default": False,
        "requires_solvability_gate": False,
        "preferred_composition_pattern": "single_family_mutation",
        "parameter_reuse_policy": "Keep the code family and equivalence notion explicit when changing block sizes or weights.",
        "deep_variant_requirement": "Prefer new stabilizer or orbit-count targets over shallow count reshuffling.",
        "preferred_repair_mode": "easier_rescue",
    },
    "graph_count": {
        "requires_research_default": False,
        "requires_briefing_default": False,
        "requires_solvability_gate": False,
        "preferred_composition_pattern": "serial_pipeline",
        "parameter_reuse_policy": "Reuse counted parameters only when the downstream graph/counting object remains explicit.",
        "deep_variant_requirement": "Prefer new admissibility or transition constraints over simple range changes.",
        "preferred_repair_mode": "easier_rescue",
    },
    "nearest_prime": {
        "requires_research_default": False,
        "requires_briefing_default": False,
        "requires_solvability_gate": False,
        "preferred_composition_pattern": "single_family_mutation",
        "parameter_reuse_policy": "Reuse constants only when the new exact filter remains explicit and verifiable.",
        "deep_variant_requirement": "Prefer exact structural filters over larger raw N.",
        "preferred_repair_mode": "easier_rescue",
    },
    "symmetric_power_sum_system": {
        "requires_research_default": False,
        "requires_briefing_default": False,
        "requires_solvability_gate": True,
        "preferred_composition_pattern": "same_system_new_parameters",
        "parameter_reuse_policy": "Prefer new solvable systems over parent-constant reuse.",
        "deep_variant_requirement": "Construct a new verified system with natural-number roots instead of coefficient shuffling.",
        "preferred_repair_mode": "easier_rescue",
    },
    "binomial_moment": {
        "requires_research_default": False,
        "requires_briefing_default": False,
        "requires_solvability_gate": False,
        "preferred_composition_pattern": "single_family_mutation",
        "parameter_reuse_policy": "Prefer changing n, p, moment order, or the exact derived target.",
        "deep_variant_requirement": "Do not use the moment only as a helper constant donor.",
        "preferred_repair_mode": "easier_rescue",
    },
    "lattice_count": {
        "requires_research_default": False,
        "requires_briefing_default": False,
        "requires_solvability_gate": False,
        "preferred_composition_pattern": "single_family_mutation",
        "parameter_reuse_policy": "Reuse combinatorial parameters only when the counting target remains explicit.",
        "deep_variant_requirement": "Prefer new counting structures or admissibility filters over simple bound changes.",
        "preferred_repair_mode": "easier_rescue",
    },
    "contour_integral": {
        "requires_research_default": False,
        "requires_briefing_default": False,
        "requires_solvability_gate": False,
        "preferred_composition_pattern": "single_family_mutation",
        "parameter_reuse_policy": "Preserve contour and analytic object definitions exactly when changing numeric parameters.",
        "deep_variant_requirement": "Prefer new exact targets or contour-preserving transformations over superficial coefficient changes.",
        "preferred_repair_mode": "easier_rescue",
    },
    "limit_integral": {
        "requires_research_default": False,
        "requires_briefing_default": False,
        "requires_solvability_gate": True,
        "preferred_composition_pattern": "serial_pipeline",
        "parameter_reuse_policy": "Preserve derived-constant consistency across statement, solution, and code.",
        "deep_variant_requirement": "Prefer exact downstream reuse of the derived constant over free coefficient drift.",
        "preferred_repair_mode": "easier_rescue",
    },
    "integer_equation_system": {
        "requires_research_default": False,
        "requires_briefing_default": False,
        "requires_solvability_gate": True,
        "preferred_composition_pattern": "same_system_new_parameters",
        "parameter_reuse_policy": "Do not reuse hidden constants when the statement defines a new equation system.",
        "deep_variant_requirement": "Keep the system satisfiable over the stated domain.",
        "preferred_repair_mode": "easier_rescue",
    },
    "derived_constant_coupling": {
        "requires_research_default": False,
        "requires_briefing_default": False,
        "requires_solvability_gate": True,
        "preferred_composition_pattern": "serial_pipeline",
        "parameter_reuse_policy": "Preserve derived-constant consistency across statement, solution, and code.",
        "deep_variant_requirement": "Use the derived constant naturally downstream instead of overriding the coupled system.",
        "preferred_repair_mode": "easier_rescue",
    },
    "generic": {
        "requires_research_default": True,
        "requires_briefing_default": True,
        "requires_solvability_gate": False,
        "preferred_composition_pattern": "serial_pipeline",
        "parameter_reuse_policy": "Avoid direct parent-constant reuse unless explicitly derived.",
        "deep_variant_requirement": "Prefer a deeper structural change over a shallow coefficient reshuffle.",
        "preferred_repair_mode": "easier_rescue",
    },
}


def infer_family_signature(problem: Dict[str, Any]) -> str:
    statement = str(problem.get("statement", "") or "")
    solution = str(problem.get("solution", "") or "")
    code = str(problem.get("code", "") or "")
    text = " ".join([statement, solution, code]).lower()
    if any(token in text for token in ["permutation matrix", "trace of", "x \\mapsto", "x \\to ax", "mod p"]) or (
        "trace(" in code.lower() and "mod" in text
    ):
        return "permutation_matrix_trace"
    if any(token in text for token in ["automorphism group", "orbit-stabilizer", "equivalence class of code", "repetition code", "codeword", "binary linear code"]):
        return "code_automorphism"
    if any(token in text for token in ["closed walk", "adjacent", "2 apart", "matrix exponentiation", "adjacency matrix", "recurrence", "graph"]) and (
        "mod" in text or "count" in text
    ):
        return "graph_count"
    if "nearest prime" in text or "isprime" in code.lower():
        return "nearest_prime"
    if "bin(" in text or "binomial" in text or "\\mathrm{bin}" in text or "moment" in text:
        return "binomial_moment"
    if "a+b+c" in text and ("newton" in text or "elementary symmetric" in text or "power sum" in text):
        return "symmetric_power_sum_system"
    if "lattice point" in text or "\\mathcal{t}" in text:
        return "lattice_count"
    if "contour" in text or "semicircular arc" in text or ("integral" in text and "complex" in text):
        return "contour_integral"
    if "integral" in text and ("lim" in text or "\\lim" in text):
        return "limit_integral"
    return "generic"


def strategy_for_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    family = problem.get("family_signature") or infer_family_signature(problem)
    return {"family_signature": family, **dict(FAMILY_STRATEGIES.get(family, FAMILY_STRATEGIES["generic"]))}


def should_use_research(item: Dict[str, Any]) -> bool:
    op_type = item.get("op_type")
    if op_type == "survivor":
        return False
    if op_type == "mutation":
        parent = ((item.get("parents") or [{}])[0]) if item.get("parents") else {}
        strategy = strategy_for_problem(parent)
        return bool(strategy.get("requires_research_default", True))
    parents = item.get("parents") or []
    parent_families = {strategy_for_problem(parent).get("family_signature", "generic") for parent in parents}
    if not parent_families or "generic" in parent_families or len(parent_families) > 1:
        return True
    return False


def should_use_llm_brief(item: Dict[str, Any], context_pack: Dict[str, Any]) -> bool:
    if item.get("op_type") == "survivor":
        return False
    if item.get("op_type") == "crossover":
        parents = item.get("parents") or []
        parent_families = {strategy_for_problem(parent).get("family_signature", "generic") for parent in parents}
        if not parent_families or "generic" in parent_families or len(parent_families) > 1:
            return True
        return False
    parent = ((item.get("parents") or [{}])[0]) if item.get("parents") else {}
    strategy = strategy_for_problem(parent)
    return bool(strategy.get("requires_briefing_default", True))


def requires_solvability_gate(problem: Dict[str, Any]) -> bool:
    strategy = strategy_for_problem(problem)
    return bool(strategy.get("requires_solvability_gate", False))
