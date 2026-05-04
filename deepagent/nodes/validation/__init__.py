"""Validation package: orchestration nodes plus shared validation worker helpers.

Phase A/B/C: `postprocess_candidates_node` and `validate_candidates_node`
were removed — their per-candidate work moved into `slot_unit_node`
(`deepagent.nodes.synthesis.slot_pipeline._run_one_postprocess` +
`validate_one_candidate`), and cross-slot decisions moved into
`slot_aggregate_node`. `ground_and_rescore_node` remains as a separate phase.
"""

from deepagent.nodes.validation.orchestrator import (  # noqa: F401
    _assess_constraint_guard as assess_constraint_guard,
    ground_and_rescore_node,
    validate_one_candidate,
)
from deepagent.nodes.validation.worker import *  # noqa: F401,F403
