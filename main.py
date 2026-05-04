import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from dotenv import load_dotenv

from artifact_views import write_all_validated_problem_views, write_latest_problem_views
from config import LANGSMITH_PROJECT, LANGSMITH_TRACING, STEADY_STATE_POOL_SIZE, validate_config
from data_paths import load_seed_problems
from deepagent.graph import create_deep_evolution_graph
from deepagent.tracing import close_slot_trace_parents, pop_manual_trace_parent, push_manual_trace_parent
from langsmith.run_helpers import tracing_context
from langsmith.run_trees import RunTree

load_dotenv()

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "data" / "runs"
logger = logging.getLogger("deep_main")
ABLATION_CONDITIONS = ("full", "no_near_copy", "no_solvability")


class _TeeStream:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _next_run_dir(base_dir: Path = RUNS_DIR, prefix: str = "deep-run") -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base_dir / f"{stamp}-{prefix}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = base_dir / f"{stamp}-{prefix}-{suffix:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _configure_logging(log_path: Path):
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.__stderr__)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler], force=True)


@contextmanager
def _tee_console(log_path: Path):
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_stdout, log_file)
    sys.stderr = _TeeStream(old_stderr, log_file)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()


def _build_run_config(thread_id: str, phase: str, generation_count: int = 0) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"{phase}_generation_{generation_count}",
        "tags": ["deepagent", phase],
        "callbacks": [],
        "recursion_limit": 300,
        "metadata": {
            "generation_count": generation_count,
            "langsmith_project": LANGSMITH_PROJECT or "unset",
        },
    }


def _canonical_target_generation_size(params: Dict[str, Any]) -> int:
    return STEADY_STATE_POOL_SIZE


def _current_generation_size(state: Dict[str, Any]) -> int:
    return int(state.get("current_generation_size") or len(state.get("current_generation", []) or []))


def _generation_trace_metadata(
    thread_id: str,
    generation_number: int,
    run_dir: Path,
    phase: str,
    seed_ids: List[str],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "thread_id": thread_id,
        "generation_number": generation_number,
        "run_dir": str(run_dir),
        "phase": phase,
        "seed_ids": seed_ids,
        "session_phase": state.get("session_phase"),
        "session_thread_id": state.get("session_thread_id"),
        "target_generation_size": _canonical_target_generation_size(state.get("parameters", {}) or {}),
        "ablation_condition": (state.get("parameters", {}) or {}).get("ablation_condition", "full"),
        "current_generation_size": _current_generation_size(state),
        "cumulative_validated_generated_count": int(state.get("cumulative_validated_generated_count", 0) or 0),
        "stop_reason": state.get("stop_reason", ""),
    }


def _start_generation_trace(
    thread_id: str,
    generation_number: int,
    run_dir: Path,
    phase: str,
    seed_ids: List[str],
    state: Dict[str, Any],
) -> Optional[RunTree]:
    if not LANGSMITH_TRACING:
        return None
    metadata = _generation_trace_metadata(thread_id, generation_number, run_dir, phase, seed_ids, state)
    root = RunTree(
        name=f"deepagent.generation.{generation_number}",
        run_type="chain",
        inputs={
            "generation_number": generation_number,
            "phase": phase,
            "state": {
                "session_phase": state.get("session_phase"),
                "generation_count": state.get("generation_count", 0),
                "awaiting_review": state.get("awaiting_review", False),
                "target_generation_size": _canonical_target_generation_size(state.get("parameters", {}) or {}),
                "current_generation_size": _current_generation_size(state),
            },
        },
        project_name=LANGSMITH_PROJECT or None,
        tags=["deepagent", "generation-root", f"generation-{generation_number}"],
        extra={"metadata": metadata},
    )
    root.post()
    return root


def _finish_generation_trace(
    generation_trace: Optional[RunTree],
    state: Dict[str, Any],
    *,
    status: str,
    error: Optional[str] = None,
) -> None:
    close_slot_trace_parents(state, default_status=status)
    if generation_trace is None:
        return
    outputs = {
        "status": status,
        "session_phase": state.get("session_phase"),
        "generation_count": state.get("generation_count", 0),
        "awaiting_review": state.get("awaiting_review", False),
        "valid_count": state.get("valid_count", 0),
        "candidate_count": len(state.get("candidates", []) or []),
        "approved_count": len(state.get("approved_candidates", []) or []),
        "failed_count": len(state.get("failed_problems", []) or []),
        "context_pack_digests": sorted((state.get("context_packs", {}) or {}).keys()),
        "target_generation_size": _canonical_target_generation_size(state.get("parameters", {}) or {}),
        "current_generation_size": _current_generation_size(state),
        "cumulative_validated_generated_count": int(state.get("cumulative_validated_generated_count", 0) or 0),
        "stop_reason": state.get("stop_reason", ""),
    }
    generation_trace.end(outputs=outputs, error=error)
    generation_trace.patch()


def _graph_trace_inputs(state: Dict[str, Any], invoke_config: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(invoke_config.get("metadata", {}) or {})
    return {
        "session_phase": state.get("session_phase", ""),
        "generation_count": state.get("generation_count", 0),
        "awaiting_review": state.get("awaiting_review", False),
        "current_generation_size": _current_generation_size(state),
        "target_generation_size": _canonical_target_generation_size(state.get("parameters", {}) or {}),
        "thread_id": metadata.get("thread_id", ""),
        "generation_root_id": metadata.get("generation_root_id", ""),
    }


def _graph_trace_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "next_phase": state.get("session_phase", ""),
        "generation_count": state.get("generation_count", 0),
        "awaiting_review": state.get("awaiting_review", False),
        "generation_handoff_ready": state.get("generation_handoff_ready", False),
        "current_generation_size": _current_generation_size(state),
        "candidate_count": len(state.get("candidates", []) or []),
        "approved_count": len(state.get("approved_candidates", []) or []),
        "failed_count": len(state.get("failed_problems", []) or []),
        "valid_count": state.get("valid_count", 0),
        "stop_reason": state.get("stop_reason", ""),
    }


def _invoke_graph(
    graph,
    state: Dict[str, Any],
    invoke_config: Dict[str, Any],
    parent_trace: Optional[RunTree],
) -> Dict[str, Any]:
    if parent_trace is None:
        return dict(graph.invoke(state, config=invoke_config))

    graph_trace = parent_trace.create_child(
        name=invoke_config.get("run_name") or "deepagent.graph",
        run_type="chain",
        inputs=_graph_trace_inputs(state, invoke_config),
        tags=list(invoke_config.get("tags", []) or []),
        extra={"metadata": dict(invoke_config.get("metadata", {}) or {})},
    )
    graph_trace.post()
    token = push_manual_trace_parent(graph_trace)
    try:
        with tracing_context(enabled=False):
            result = dict(graph.invoke(state, config=invoke_config))
        graph_trace.end(outputs=_graph_trace_outputs(result))
        return result
    except Exception as exc:
        graph_trace.end(
            outputs={"status": "error", "next_phase": state.get("session_phase", "")},
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        pop_manual_trace_parent(token)
        graph_trace.patch()


def _parse_review_action(raw: str) -> tuple[str, dict]:
    choice = (raw or "").strip()
    normalized = choice.lower()
    if normalized.startswith("c"):
        return "continue", {}
    if normalized in {"exit", "quit", "stop"} or normalized.startswith("e"):
        return "exit", {}
    if normalized.startswith("advisor") or normalized.startswith("advice"):
        request = choice.split(":", 1)[1].strip() if ":" in choice else "Analyze generation and suggest next-step parameter updates."
        return "advisor", {"request": request}
    if normalized.startswith("modify"):
        payload = choice[len("modify"):].strip()
        if not payload:
            payload = input("Enter '<problem_id>: <request>': ").strip()
        if ":" not in payload:
            return "", {}
        problem_id, request = payload.split(":", 1)
        return "modify", {"problem_id": problem_id.strip(), "request": request.strip()}
    return "", {}


def _resolve_seed_ids(seed_ids_arg: str) -> List[str]:
    return [part.strip() for part in (seed_ids_arg or "").split(",") if part.strip()]


def _select_seed_generation(seed_spec: Optional[str], seed_ids: List[str]) -> List[Dict]:
    seeds = load_seed_problems(seed_spec)
    if not seed_ids:
        if not seeds:
            raise RuntimeError("Closed-loop runs require at least one seed problem.")
        return [dict(problem) for problem in seeds]

    by_id = {problem.get("id"): problem for problem in seeds}
    selected = []
    for seed_id in seed_ids:
        if seed_id not in by_id:
            raise RuntimeError(f"Unknown seed id: {seed_id}")
        selected.append(dict(by_id[seed_id]))
    if not selected:
        raise RuntimeError("Closed-loop runs require at least one selected seed ID.")
    return selected


def _final_state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "session_phase": state.get("session_phase"),
        "awaiting_review": state.get("awaiting_review", False),
        "generation_count": state.get("generation_count", 0),
        "current_generation_size": _current_generation_size(state),
        "target_generation_size": _canonical_target_generation_size(state.get("parameters", {}) or {}),
        "cumulative_validated_generated_count": int(state.get("cumulative_validated_generated_count", 0) or 0),
        "stop_reason": state.get("stop_reason", ""),
        "current_generation": state.get("current_generation", []),
        "candidates": state.get("candidates", []),
        "approved_candidates": state.get("approved_candidates", []),
        "failed_problems": state.get("failed_problems", []),
        "research_artifacts": state.get("research_artifacts", []),
        "validation_feedback": state.get("validation_feedback", []),
        "review_prompt": state.get("review_prompt", ""),
        "pair_results": state.get("pair_results", {}),
        "pair_health_scores": state.get("pair_health_scores", {}),
        "mutation_policies": state.get("mutation_policies", {}),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DeepAgent closed-loop generation pipeline.")
    parser.add_argument("--seed-file", default=os.environ.get("DEEP_SEED_FILE", "").strip() or None)
    parser.add_argument("--seed-ids", default=os.environ.get("DEEP_SEED_IDS", "").strip())
    parser.add_argument("--use-latest", action="store_true", default=os.environ.get("USE_LATEST_GEN", "0") == "1")
    parser.add_argument("--max-generations", type=int, default=int(os.environ.get("DEEP_MAX_GENERATIONS") or os.environ.get("MAX_GENERATIONS") or "1"))
    parser.add_argument(
        "--target-generation-size",
        type=int,
        default=int(os.environ.get("DEEP_TARGET_GENERATION_SIZE") or os.environ.get("TARGET_GENERATION_SIZE") or "5"),
    )
    parser.add_argument(
        "--target-problem-count",
        type=int,
        default=int(os.environ.get("DEEP_TARGET_PROBLEM_COUNT") or os.environ.get("TARGET_PROBLEM_COUNT") or "0") or None,
    )
    parser.add_argument("--max-parallel-dispatch", type=int, default=int(os.environ.get("DEEP_MAX_PARALLEL_DISPATCH") or "4"))
    parser.add_argument(
        "--ablation-condition",
        choices=ABLATION_CONDITIONS,
        default=os.environ.get("DEEP_ABLATION_CONDITION", "full").strip() or "full",
        help="Validation-gate ablation condition for micro-study runs.",
    )
    parser.add_argument("--run-dir", default=os.environ.get("DEEP_RUN_DIR", "").strip() or None)
    parser.add_argument("--save-format", default=os.environ.get("DEEP_SAVE_FORMAT", "").strip() or None)
    parser.add_argument("--artifact-dir", default=os.environ.get("DEEP_ARTIFACT_DIR", "").strip() or None)
    parser.add_argument("--interactive-review", action="store_true", default=False)
    parser.add_argument("--mutation-only", action="store_true", default=os.environ.get("MUTATION_ONLY", "0") == "1" or os.environ.get("DEEP_MUTATION_ONLY", "0") == "1")
    parser.add_argument(
        "--min-survivable-population",
        type=int,
        default=int(os.environ.get("DEEP_MIN_SURVIVABLE_POPULATION", "0") or "0") or None,
        help="Minimum slots that must pass validation for a generation to save. Defaults to DEFAULT_PARAMS['min_survivable_population'] (3).",
    )
    parser.add_argument(
        "--max-slot-regen-attempts",
        type=int,
        default=int(os.environ.get("DEEP_MAX_SLOT_REGEN_ATTEMPTS", "0") or "0") or None,
        help="How many times an individual slot can be regenerated after a validation failure. Defaults to 3.",
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()
    errors, warnings = validate_config()
    if errors:
        raise RuntimeError("Configuration errors:\n- " + "\n- ".join(errors))

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _next_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_path = run_dir / "runtime.log"
    step_log_path = run_dir / "session.log"
    _configure_logging(step_log_path)

    with _tee_console(runtime_log_path):
        for warning in warnings:
            print(f"[CONFIG] Warning: {warning}")
        if LANGSMITH_TRACING:
            print(f"[TRACE] LangSmith tracing enabled for project '{LANGSMITH_PROJECT or 'default'}'.")

        if args.use_latest:
            current_generation = None
            selected_seed_ids = []
        else:
            selected = _select_seed_generation(args.seed_file, _resolve_seed_ids(args.seed_ids))
            current_generation = selected
            selected_seed_ids = [problem.get("id") for problem in selected]

        save_format_path = Path(args.save_format).expanduser().resolve() if args.save_format else (run_dir / "generation_{gen}.json")
        artifact_dir_path = Path(args.artifact_dir).expanduser().resolve() if args.artifact_dir else run_dir
        artifact_dir_path.mkdir(parents=True, exist_ok=True)
        if int(args.target_generation_size) != STEADY_STATE_POOL_SIZE:
            logger.info(
                "Ignoring requested target_generation_size=%s; steady-state pool size is fixed at %s.",
                args.target_generation_size,
                STEADY_STATE_POOL_SIZE,
            )

        initial_state = {
            "session_initialized": False,
            "session_phase": "load_or_resume",
            "session_thread_id": f"deep-pipeline-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "current_generation": current_generation,
            "current_generation_size": len(current_generation or []),
            "generation_count": 0,
            "desired_generation_size": STEADY_STATE_POOL_SIZE,
            "target_generation_size": STEADY_STATE_POOL_SIZE,
            "cumulative_validated_generated_count": 0,
            "stop_reason": "",
            "session_options": {
                "seed_spec": args.seed_file,
                "use_latest": args.use_latest,
                "save_format": str(save_format_path),
                "artifact_dir": str(artifact_dir_path),
                "mutation_only": args.mutation_only,
                "max_generations": args.max_generations,
                "target_generation_size": STEADY_STATE_POOL_SIZE,
                "target_problem_count": args.target_problem_count,
                "interactive_review": args.interactive_review,
                "ablation_condition": args.ablation_condition,
            },
            "parameters": {
                "population_size": STEADY_STATE_POOL_SIZE,
                "target_generation_size": STEADY_STATE_POOL_SIZE,
                "max_parallel_dispatch": args.max_parallel_dispatch,
                "auto_continue_generations": not args.interactive_review,
                "require_all_valid": True,
                "interactive_review": args.interactive_review,
                "ablation_condition": args.ablation_condition,
                **({"min_survivable_population": int(args.min_survivable_population)} if args.min_survivable_population else {}),
                **({"max_slot_regen_attempts": int(args.max_slot_regen_attempts)} if args.max_slot_regen_attempts else {}),
            },
            "review_action": "",
            "review_payload": {},
        }

        graph = create_deep_evolution_graph()
        state = initial_state
        thread_id = initial_state["session_thread_id"]
        active_generation_trace: Optional[RunTree] = None
        active_generation_number: Optional[int] = None
        summary: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "log_files": {
                "runtime": str(runtime_log_path),
                "session": str(step_log_path),
            },
            "seed_ids": selected_seed_ids,
            "parameters": initial_state["parameters"],
            "session_options": initial_state["session_options"],
            "generation_runs": [],
            "steps": [],
        }

        try:
            while True:
                if state.get("should_exit") or state.get("session_phase") == "exit":
                    print("[DEEP] Exiting session.")
                    break

                if state.get("session_phase") == "generation_handoff_ready":
                    max_generations = (state.get("session_options", {}) or {}).get("max_generations")
                    if max_generations is not None and state.get("generation_count", 0) >= max_generations:
                        state["session_phase"] = "exit"
                        state["should_exit"] = True
                        continue
                    state["session_phase"] = "init_run_memory"
                    state["generation_handoff_ready"] = False
                    state["review_action"] = ""
                    state["review_payload"] = {}
                    continue

                if state.get("awaiting_review"):
                    current_generation = state.get("current_generation", []) or []
                    if current_generation:
                        print("[DEEP] Generated problems with difficulty:")
                        for idx, problem in enumerate(current_generation, 1):
                            print(f"  - {problem.get('id', f'prob_{idx}')}: difficulty={problem.get('difficulty', 'unknown')}")
                    print(state.get("review_prompt", "Generation ready for review."))
                    if args.interactive_review:
                        action, payload = _parse_review_action(input("Next action? ").strip())
                        if not action:
                            print("[DEEP] Unknown command. Use continue, advisor[: request], modify <id>: <request>, or exit.")
                            continue
                    else:
                        action = "exit"
                        payload = {}
                    state["session_phase"] = "review_generation"
                    state["awaiting_review"] = False
                    state["generation_handoff_ready"] = False
                    state["review_action"] = action
                    state["review_payload"] = payload
                    invoke_config = {
                        **_build_run_config(thread_id, "review_generation", state.get("generation_count", 0)),
                        "run_name": f"deepagent.review.{state.get('generation_count', 0)}",
                        "tags": ["deepagent", "review"],
                        "metadata": {
                            "generation_count": state.get("generation_count", 0),
                            "thread_id": thread_id,
                            "generation_root_name": f"deepagent.generation.{active_generation_number or state.get('generation_count', 0)}",
                            "generation_root_id": str(active_generation_trace.id) if active_generation_trace else "",
                        },
                    }
                    state = _invoke_graph(graph, state, invoke_config, active_generation_trace)
                    if active_generation_trace is not None and (
                        state.get("session_phase") in {"generation_handoff_ready", "exit"}
                    ):
                        _finish_generation_trace(
                            active_generation_trace,
                            state,
                            status="completed" if state.get("session_phase") != "exit" else "exited",
                        )
                        active_generation_trace = None
                        active_generation_number = None
                    summary["steps"].append(
                        {
                            "phase": state.get("session_phase"),
                            "awaiting_review": state.get("awaiting_review", False),
                            "generation_count": state.get("generation_count", 0),
                            "current_generation_size": _current_generation_size(state),
                            "target_generation_size": _canonical_target_generation_size(state.get("parameters", {}) or {}),
                            "cumulative_validated_generated_count": int(state.get("cumulative_validated_generated_count", 0) or 0),
                            "stop_reason": state.get("stop_reason", ""),
                            "valid_count": state.get("valid_count", 0),
                            "candidate_count": len(state.get("candidates", []) or []),
                            "approved_count": len(state.get("approved_candidates", []) or []),
                            "failed_count": len(state.get("failed_problems", []) or []),
                            "trace_scope": "review",
                        }
                    )
                    continue

                start_generation = state.get("generation_count", 0)
                target_generation = start_generation + 1
                if active_generation_trace is None or active_generation_number != target_generation:
                    if active_generation_trace is not None:
                        _finish_generation_trace(active_generation_trace, state, status="rolled_forward")
                    active_generation_trace = _start_generation_trace(
                        thread_id,
                        target_generation,
                        run_dir,
                        state.get("session_phase") or "session",
                        selected_seed_ids,
                        state,
                    )
                    active_generation_number = target_generation
                generation_record = {
                    "generation": target_generation,
                    "start_phase": state.get("session_phase"),
                    "start_generation_count": start_generation,
                }
                phase = state.get("session_phase") or "session"
                invoke_config = {
                    **_build_run_config(thread_id, phase, start_generation),
                    "run_name": f"deepagent.generation.{target_generation}.graph",
                    "tags": ["deepagent", "generation", phase],
                        "metadata": {
                            "generation_count": target_generation,
                            "thread_id": thread_id,
                            "seed_ids": selected_seed_ids,
                            "run_dir": str(run_dir),
                            "generation_root_name": f"deepagent.generation.{target_generation}",
                            "generation_root_id": str(active_generation_trace.id) if active_generation_trace else "",
                            "target_generation_size": _canonical_target_generation_size(state.get("parameters", {}) or {}),
                        },
                    }
                state = _invoke_graph(graph, state, invoke_config, active_generation_trace)
                if active_generation_trace is not None and (
                    (state.get("session_phase") in {"generation_handoff_ready", "exit"})
                    or (state.get("generation_count", 0) > target_generation)
                ):
                    _finish_generation_trace(
                        active_generation_trace,
                        state,
                        status="completed" if state.get("session_phase") != "exit" else "exited",
                    )
                    active_generation_trace = None
                    active_generation_number = None
                summary["steps"].append(
                    {
                        "phase": state.get("session_phase"),
                        "awaiting_review": state.get("awaiting_review", False),
                        "generation_count": state.get("generation_count", 0),
                        "current_generation_size": _current_generation_size(state),
                        "target_generation_size": _canonical_target_generation_size(state.get("parameters", {}) or {}),
                        "cumulative_validated_generated_count": int(state.get("cumulative_validated_generated_count", 0) or 0),
                        "stop_reason": state.get("stop_reason", ""),
                        "valid_count": state.get("valid_count", 0),
                        "candidate_count": len(state.get("candidates", []) or []),
                        "approved_count": len(state.get("approved_candidates", []) or []),
                        "failed_count": len(state.get("failed_problems", []) or []),
                        "trace_scope": f"generation_{target_generation}",
                    }
                )
                generation_record.update(
                    {
                        "end_phase": state.get("session_phase"),
                        "end_generation_count": state.get("generation_count", 0),
                        "awaiting_review": state.get("awaiting_review", False),
                        "failed_count": len(state.get("failed_problems", []) or []),
                        "current_generation_size": _current_generation_size(state),
                        "cumulative_validated_generated_count": int(state.get("cumulative_validated_generated_count", 0) or 0),
                        "stop_reason": state.get("stop_reason", ""),
                    }
                )
                summary["generation_runs"].append(generation_record)

        except Exception as exc:
            logger.exception("DeepAgent main run failed")
            _finish_generation_trace(active_generation_trace, state, status="error", error=str(exc))
            active_generation_trace = None
            summary["error"] = {"type": type(exc).__name__, "message": str(exc)}

        _finish_generation_trace(active_generation_trace, state, status="completed" if not summary.get("error") else "error")
        latest_view = write_latest_problem_views(run_dir)
        validated_views = write_all_validated_problem_views(run_dir)
        summary["exports"] = {
            "latest_problem_markdown": str(latest_view) if latest_view else "",
            "validated_problem_markdowns": [str(path) for path in validated_views],
        }
        summary["final_state"] = _final_state_snapshot(state)
        (run_dir / "full_run_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DEEP] Run artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
