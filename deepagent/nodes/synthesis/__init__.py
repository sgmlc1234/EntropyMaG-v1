"""Synthesis-phase node package.

Phase A/B/C: legacy nodes (`synthesize_candidates_node`,
`repair_failed_candidates_node`) were removed in favor of slot fan-out
(`slot_dispatch_node`, `slot_unit_node`, `slot_aggregate_node` in
`deepagent.nodes.synthesis.slot_pipeline`). The `orchestrator` module
retains shared helpers (`_run_repair_item`, `_attach_peer_context`,
`_build_synthesis_dispatch_args`) used by the slot_pipeline.
"""

