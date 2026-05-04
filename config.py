"""
Configuration module for API keys and settings.
Loads from environment variables with fallback to .env file.
Uses OpenRouter API exclusively (NOT OpenAI API directly).
"""
import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# API Keys (OpenRouter only)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes", "on"}
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "").strip()
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "").strip()
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "").strip()

if LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_CALLBACKS_BACKGROUND", "false")
    if LANGSMITH_ENDPOINT:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", LANGSMITH_ENDPOINT)
    if LANGSMITH_PROJECT:
        os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
    if LANGSMITH_API_KEY:
        os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)

# OpenRouter API Base URL (NOT OpenAI)
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
STEADY_STATE_POOL_SIZE = 5
PYTHON_SANDBOX_MAX_CONCURRENCY = max(1, int(os.getenv("DEEPAGENT_SANDBOX_MAX_CONCURRENCY", "4") or "4"))


# Model configurations.
#
# Agents group into two roles:
#   - Orchestrator / control tower: plan, dispatch, validate, review. Benefits
#     from a smarter model because it composes other agents' work. All
#     orchestrator-role agents share the same MODELS["orchestrator"] model by
#     default; override per-agent below if needed.
#   - Worker: researcher and generator actually emit long structured content.
#     These stay on the cheaper/faster model unless USE_LATE_MODELS is set.
#
# Note on consolidation: mutation and crossover share a single generator LLM
# (see deepagent/generator.py); "mutator" / "crossover" entries are retained
# only as legacy aliases in case downstream scripts still look them up.
ORCHESTRATOR_MODEL = "google/gemini-3.1-flash-lite-preview"
WORKER_MODEL = "google/gemini-3.1-flash-lite-preview"

MODELS = {
    # Orchestrator / control tower
    "orchestrator": ORCHESTRATOR_MODEL,
    "selector": ORCHESTRATOR_MODEL,
    "synthesis_planner": ORCHESTRATOR_MODEL,
    "regen_planner": ORCHESTRATOR_MODEL,
    "brief_validator": ORCHESTRATOR_MODEL,
    "advisor": ORCHESTRATOR_MODEL,
    "validator": ORCHESTRATOR_MODEL,
    "modifier": ORCHESTRATOR_MODEL,
    # Worker
    "generator": WORKER_MODEL,         # unified mutation + crossover generator
    "researcher": WORKER_MODEL,
    "quality": WORKER_MODEL,
    "hotfix": WORKER_MODEL,
    "code_generator": WORKER_MODEL,
    # Legacy aliases (kept for backwards compatibility; prefer "generator"):
    "mutator": WORKER_MODEL,
    "crossover": WORKER_MODEL,
}

# Worker agents that use WORKER_MODEL.
# reasoning_budget/thinking_budget=0 is restored for generator-operation
# agents because long structured generations are hitting provider completion
# ceilings and surfacing LengthFinishReasonError.
_WORKER_AGENTS: frozenset = frozenset({
    "generator", "researcher", "quality", "hotfix", "code_generator",
    "mutator", "crossover",  # legacy aliases
})
_GENERATOR_REASONING_DISABLED: dict = {
    "reasoning_budget": 0,
    "thinking_budget": 0,
}
_WORKER_EXTRA_BODY: dict = dict(_GENERATOR_REASONING_DISABLED)

# Optional late-stage models (e.g., GPT-4/5 class) when USE_LATE_MODELS=1.
# Applied to workers that emit the actual problem JSON; the orchestrator stays
# on its flagship model regardless.
LATE_MODELS = {
    "generator": "openai/gpt-5.4-mini",
    "mutator": "openai/gpt-5.4-mini",
    "crossover": "openai/gpt-5.4-mini",
}


# Default parameters


DEFAULT_PARAMS = {
    "easy_ratio": 0.8,  # 80% chance of easy mutation/crossover
    "population_size": STEADY_STATE_POOL_SIZE,  # Deprecated alias for target_generation_size
    "target_generation_size": STEADY_STATE_POOL_SIZE,  # Desired steady-state problems per generation
    "mutation_only": False,  # When true, mutate all seeds and skip crossover
    "require_all_valid": True,
    "auto_continue_generations": True,
    "max_validation_retries": 2,
    "max_parallel_dispatch": 4,
    "ablation_condition": "full",
    "max_slot_regen_attempts": 4,  # Total attempts per slot (1 synth + up to 3 repairs); matches _slot_unit_body default.
    "max_research_refetch": 1,  # Per-slot cap on how many times the regen_planner can route a slot back through the researcher.
    "min_survivable_population": 2,
}



# Per-agent output token ceilings. Tuned so (a) no agent requests more than the
# underlying model can return in a single call, avoiding provider-side truncation
# that surfaces as LengthFinishReasonError during structured-output parsing, and
# (b) agents that emit large structured payloads (full problem statements +
# verification code) have enough headroom.
#
# Reference ceilings (approximate, as of 2026-04):
#   - google/gemini-3.1-flash-lite-preview : output cap ~65K tokens
#   - google/gemini-3-flash-preview        : output cap ~65K tokens
#   - openai/gpt-5.4-mini                  : output cap ~128K tokens
# Values are intentionally set above the model cap so the provider never
# silently truncates due to a too-low max_tokens request; the model's own
# ceiling is the effective hard limit.
AGENT_MAX_TOKENS = {
    # Orchestrator / control-tower agents — set to 1M so the provider-side
    # ceiling is the only hard limit (Gemini flash: ~65K; GPT class: ~128K).
    "selector": 1_000_000,
    "advisor": 1_000_000,
    "modifier": 1_000_000,
    "validator": 1_000_000,         # regenerability + equivalence checks
    "briefing": 1_000_000,          # synthesis brief JSON
    "synthesis_planner": 1_000_000, # synthesis plan JSON
    "regen_planner": 1_000_000,     # per-slot retry decisions
    "brief_validator": 1_000_000,   # plan-overlaid brief
    "modifier_stage": 1_000_000,
    # Worker agents — 480K is more than sufficient for structured output
    "generator": 480000,         # unified mutation + crossover worker
    "mutator": 480000,           # legacy alias — mirrors generator
    "crossover": 480000,         # legacy alias — mirrors generator
    "code_generator": 480000,
    "researcher": 480000,        # research artifact + source summaries
    "quality": 480000,           # quality advisory flags
    "repair": 480000,            # full repaired problem
    "hotfix": 480000,            # code-only / target-only hotfix
}
DEFAULT_AGENT_MAX_TOKENS = 480000


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_llm_config(agent_name: str) -> dict:
    """Get LLM configuration for a specific agent using OpenRouter API."""
    use_late = os.getenv("USE_LATE_MODELS", "0") == "1"
    model = MODELS.get(agent_name, "google/gemini-3.1-flash-lite-preview")
    if use_late:
        model = LATE_MODELS.get(agent_name, model)
    max_tokens = AGENT_MAX_TOKENS.get(agent_name, DEFAULT_AGENT_MAX_TOKENS)
    max_tokens = _env_int("ENTROPYMATH_AGENT_MAX_TOKENS", max_tokens)
    max_tokens = _env_int(f"ENTROPYMATH_{agent_name.upper()}_MAX_TOKENS", max_tokens)
    timeout_seconds = int(os.getenv("ENTROPYMATH_LLM_TIMEOUT_SECONDS", "300") or "300")
    max_retries = int(os.getenv("ENTROPYMATH_LLM_MAX_RETRIES", "2") or "2")
    cfg = {
        "model": model,
        "api_key": OPENROUTER_API_KEY,
        "base_url": OPENROUTER_API_BASE,
        "timeout": timeout_seconds,
        "max_retries": max_retries,
        "max_tokens": max_tokens,
    }
    if agent_name in _WORKER_AGENTS:
        cfg["extra_body"] = _WORKER_EXTRA_BODY
    return cfg


def validate_config():
    """Validate that required configuration is present."""
    errors = []
    warnings = []
    
    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY is not set")
    
    if not TAVILY_API_KEY:
        warnings.append("TAVILY_API_KEY is not set (web search will not work)")

    if LANGSMITH_TRACING:
        if not LANGSMITH_API_KEY:
            errors.append("LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set")
        if not LANGSMITH_PROJECT:
            warnings.append("LANGSMITH_TRACING is enabled but LANGSMITH_PROJECT is not set")

    return errors, warnings
