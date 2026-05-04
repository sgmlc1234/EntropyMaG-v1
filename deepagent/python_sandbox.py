import ast
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from langsmith.run_helpers import get_current_run_tree

from config import PYTHON_SANDBOX_MAX_CONCURRENCY

RUNTIME_MODE = Literal["numeric_python", "scientific_python", "symbolic_python"]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

COMMON_ALLOWED_IMPORTS = {
    "math",
    "random",
    "cmath",
    "fractions",
    "decimal",
    "statistics",
    "itertools",
    "collections",
    "functools",
    "operator",
    "re",
    "bisect",
    "heapq",
}
SCIENTIFIC_ALLOWED_IMPORTS = COMMON_ALLOWED_IMPORTS | {"numpy", "scipy", "mpmath", "networkx"}
SYMBOLIC_ALLOWED_IMPORTS = SCIENTIFIC_ALLOWED_IMPORTS | {"sympy"}

BANNED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "httpx",
    "urllib",
    "shutil",
    "pathlib",
    "asyncio",
    "multiprocessing",
    "threading",
    "ctypes",
}

BANNED_CALL_NAMES = {"eval", "exec", "compile", "__import__", "open", "input"}

MODE_TIMEOUTS = {
    "numeric_python": 300,
    "scientific_python": 300,
    "symbolic_python": 300,
}

_PYTHON_SANDBOX_SEMAPHORE = threading.BoundedSemaphore(PYTHON_SANDBOX_MAX_CONCURRENCY)


def sandbox_python_bin() -> str:
    configured = os.getenv("DEEPAGENT_PYTHON_BIN", "").strip()
    if configured:
        return configured
    if DEFAULT_VENV_PYTHON.exists():
        return str(DEFAULT_VENV_PYTHON)
    return sys.executable


def detect_runtime_mode(code: str) -> RUNTIME_MODE:
    imports = extract_imports(code)
    if "sympy" in imports:
        return "symbolic_python"
    if any(name in imports for name in {"numpy", "scipy", "mpmath", "networkx"}):
        return "scientific_python"
    return "numeric_python"


def allowed_imports_for_mode(mode: RUNTIME_MODE) -> set[str]:
    if mode == "symbolic_python":
        return SYMBOLIC_ALLOWED_IMPORTS
    if mode == "scientific_python":
        return SCIENTIFIC_ALLOWED_IMPORTS
    return COMMON_ALLOWED_IMPORTS


def extract_imports(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return imports


def preflight_python_code(code: str, mode: RUNTIME_MODE) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "ok": False,
            "error_type": "syntax_error",
            "message": f"{exc.__class__.__name__}: {exc.msg} (line {exc.lineno})",
            "imports": [],
            "mode": mode,
        }

    imports = extract_imports(code)
    allowed_imports = allowed_imports_for_mode(mode)
    for imported in imports:
        if imported in BANNED_IMPORTS:
            return {
                "ok": False,
                "error_type": "import_not_allowed",
                "message": f"Import '{imported}' is not allowed in {mode}.",
                "imports": imports,
                "mode": mode,
            }
        if imported not in allowed_imports:
            return {
                "ok": False,
                "error_type": "import_not_allowed",
                "message": f"Import '{imported}' is not supported in {mode}. Allowed imports: {', '.join(sorted(allowed_imports))}.",
                "imports": imports,
                "mode": mode,
            }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BANNED_CALL_NAMES:
                return {
                    "ok": False,
                    "error_type": "unsafe_call",
                    "message": f"Call '{func.id}(...)' is not allowed in verification code.",
                    "imports": imports,
                    "mode": mode,
                }
    return {"ok": True, "imports": imports, "mode": mode}


def _sandbox_prelude(imports: List[str]) -> str:
    lines = ["import math as _deepagent_math"]
    if "numpy" in set(imports or []):
        lines.extend(
            [
                "import numpy as _deepagent_numpy",
                "if not hasattr(_deepagent_numpy, 'math'):",
                "    _deepagent_numpy.math = _deepagent_math",
            ]
        )
    return "\n".join(lines) + "\n"


def _normalize_traceback(stderr: str) -> str:
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if any("Traceback" in line for line in lines):
        return lines[-1]
    return lines[0]


def _classify_runtime_error(stderr: str) -> str:
    text = _normalize_traceback(stderr)
    lowered = text.lower()
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        return "dependency_missing"
    if "syntaxerror" in lowered:
        return "syntax_error"
    if "nameerror" in lowered:
        return "name_error"
    if "typeerror" in lowered:
        if "residue(" in lowered or "missing 1 required positional argument" in lowered:
            return "api_misuse"
        return "type_error"
    if "attributeerror" in lowered:
        return "api_misuse"
    if "valueerror" in lowered:
        return "value_error"
    if "assertionerror" in lowered:
        return "assertion_error"
    return "runtime_error"


def execute_python_code(code: str, mode: Optional[RUNTIME_MODE] = None, trace_enabled: bool = True) -> Dict[str, Any]:
    runtime_mode = mode or detect_runtime_mode(code)
    preflight = preflight_python_code(code, runtime_mode)
    if not preflight.get("ok"):
        return {
            "ok": False,
            "mode": runtime_mode,
            "python_bin": sandbox_python_bin(),
            "stdout": "",
            "stderr": preflight["message"],
            "returncode": -1,
            "error_type": preflight["error_type"],
            "error_message": preflight["message"],
            "imports": preflight.get("imports", []),
            "timeout_seconds": MODE_TIMEOUTS[runtime_mode],
        }

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("PYTHONHASHSEED", "0")
    timeout_seconds = MODE_TIMEOUTS[runtime_mode]
    python_bin = sandbox_python_bin()
    sandbox_code = _sandbox_prelude(preflight.get("imports", [])) + code
    run = None
    if trace_enabled:
        parent = get_current_run_tree()
        if parent is not None:
            run = parent.create_child(
                name="deepagent.python_sandbox",
                run_type="tool",
                inputs={
                    "mode": runtime_mode,
                    "imports": preflight.get("imports", []),
                    "python_bin": python_bin,
                    "timeout_seconds": timeout_seconds,
                },
                tags=["deepagent", "python-sandbox", runtime_mode],
                extra={
                    "metadata": {
                        "sandbox_mode": runtime_mode,
                        "sandbox_python_bin": python_bin,
                        "sandbox_timeout_seconds": timeout_seconds,
                    }
                },
            )
            run.post()

    try:
        try:
            with _PYTHON_SANDBOX_SEMAPHORE:
                result = subprocess.run(
                    [python_bin, "-I", "-c", sandbox_code],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=str(REPO_ROOT),
                    env=env,
                )
        except subprocess.TimeoutExpired:
            outcome = {
                "ok": False,
                "mode": runtime_mode,
                "python_bin": python_bin,
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "error_type": "timeout",
                "error_message": f"Code execution timed out after {timeout_seconds} seconds.",
                "imports": preflight.get("imports", []),
                "timeout_seconds": timeout_seconds,
            }
            if run is not None:
                run.end(outputs=outcome)
            return outcome
        except Exception as exc:
            outcome = {
                "ok": False,
                "mode": runtime_mode,
                "python_bin": python_bin,
                "stdout": "",
                "stderr": str(exc),
                "returncode": -1,
                "error_type": "runtime_error",
                "error_message": f"Sandbox execution failed: {exc}",
                "imports": preflight.get("imports", []),
                "timeout_seconds": timeout_seconds,
            }
            if run is not None:
                run.end(outputs=outcome, error=str(exc))
            return outcome

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            error_type = _classify_runtime_error(stderr)
            error_message = _normalize_traceback(stderr) or "Unknown execution error"
            outcome = {
                "ok": False,
                "mode": runtime_mode,
                "python_bin": python_bin,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
                "error_type": error_type,
                "error_message": error_message[:400],
                "imports": preflight.get("imports", []),
                "timeout_seconds": timeout_seconds,
            }
            if run is not None:
                run.end(outputs=outcome)
            return outcome

        outcome = {
            "ok": True,
            "mode": runtime_mode,
            "python_bin": python_bin,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
            "error_type": "",
            "error_message": "",
            "imports": preflight.get("imports", []),
            "timeout_seconds": timeout_seconds,
        }
        if run is not None:
            run.end(outputs=outcome)
        return outcome
    finally:
        if run is not None:
            run.patch()
