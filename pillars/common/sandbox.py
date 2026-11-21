"""AST safety verification, EVOLVE-BLOCK splicing, and timeout-guarded execution."""

from __future__ import annotations

import ast
import concurrent.futures
import difflib
import signal
import threading
import types
from typing import Any, Callable

from pillars.common.models import EVOLVE_END_MARKER, EVOLVE_START_MARKER


class SandboxViolationError(ValueError):
    """Raised when candidate source violates EVOLVE-BLOCK, AST, or timeout guardrails."""


class CandidateTimeoutError(TimeoutError):
    """Raised when candidate evaluation exceeds the allowed wall-clock budget."""


FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "ctypes",
        "urllib",
        "http",
        "requests",
        "httpx",
        "pickle",
        "marshal",
        "importlib",
        "inspect",
        "builtins",
        " multiprocessing",
        "multiprocessing",
        "threading",
        "signal",
        " tempfile",
        "tempfile",
        "glob",
    }
)

FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "input",
        "breakpoint",
    }
)

FORBIDDEN_DUNDER_ATTRS: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__bases__",
        "__globals__",
        "__code__",
        "__closure__",
        "__builtins__",
        "__dict__",
        "__class__",
    }
)


def extract_evolve_block(source_code: str, *, require_markers: bool = True) -> str:
    """Extract mutable code inside strict column-0 EVOLVE-BLOCK markers.

    Enforces AlphaEvolve formatting rules:
    - Markers must start at column 0 with no inline trailing text.
    - Block must contain at least one executable AST statement (not empty or comment-only).
    """
    lines = source_code.splitlines()
    start_indices: list[int] = []
    end_indices: list[int] = []

    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if EVOLVE_START_MARKER in raw_line:
            if not raw_line.startswith(EVOLVE_START_MARKER) or stripped != EVOLVE_START_MARKER:
                raise SandboxViolationError(
                    f"Line {idx + 1}: '{EVOLVE_START_MARKER}' must start at column 0 with no inline text."
                )
            start_indices.append(idx)
        elif EVOLVE_END_MARKER in raw_line:
            if not raw_line.startswith(EVOLVE_END_MARKER) or stripped != EVOLVE_END_MARKER:
                raise SandboxViolationError(
                    f"Line {idx + 1}: '{EVOLVE_END_MARKER}' must start at column 0 with no inline text."
                )
            end_indices.append(idx)

    if not start_indices and not end_indices:
        if require_markers:
            raise SandboxViolationError(
                f"Candidate is missing required '{EVOLVE_START_MARKER}' and '{EVOLVE_END_MARKER}' markers."
            )
        block_text = source_code
    else:
        if len(start_indices) != 1 or len(end_indices) != 1:
            raise SandboxViolationError(
                f"Expected exactly one EVOLVE-BLOCK pair, found {len(start_indices)} start and {len(end_indices)} end markers."
            )
        s_idx, e_idx = start_indices[0], end_indices[0]
        if e_idx <= s_idx + 1:
            raise SandboxViolationError("EVOLVE-BLOCK is empty.")
        block_lines = lines[s_idx + 1 : e_idx]
        block_text = "\n".join(block_lines)

    if not block_text.strip():
        raise SandboxViolationError("EVOLVE-BLOCK contains only whitespace.")

    # Verify non-comment executable Python statements inside block
    # (supports both top-level column-0 blocks and indented function-body blocks with raw multiline strings)
    parsed_tree: ast.Module | None = None
    last_syn_err: SyntaxError | None = None
    for candidate_ast_src in (block_text, f"def _ae_block_scope():\n{block_text}\n"):
        try:
            parsed_tree = ast.parse(candidate_ast_src)
            break
        except SyntaxError as exc:
            last_syn_err = exc

    if parsed_tree is None:
        raise SandboxViolationError(f"Syntax error inside EVOLVE-BLOCK: {last_syn_err}") from last_syn_err

    stmts = (
        parsed_tree.body[0].body  # type: ignore[attr-defined]
        if len(parsed_tree.body) == 1
        and isinstance(parsed_tree.body[0], ast.FunctionDef)
        and parsed_tree.body[0].name == "_ae_block_scope"
        else parsed_tree.body
    )
    if not stmts or (
        len(stmts) == 1 and isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant)
    ):
        raise SandboxViolationError(
            "EVOLVE-BLOCK contains only comments/docstrings and zero executable Python statements."
        )

    return block_text


def _indent_col0_block_preserving_multiline_strings(code_lines: list[str], prefix: str) -> list[str]:
    """Indent column-0 Python statements by `prefix` without corrupting multiline string interiors."""
    joined = "\n".join(code_lines)
    try:
        tree = ast.parse(joined)
    except SyntaxError:
        return [f"{prefix}{ln}" if ln.strip() else ln for ln in code_lines]

    interior_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            start_ln = getattr(node, "lineno", 0)
            end_ln = getattr(node, "end_lineno", 0)
            if end_ln > start_ln > 0:
                for ln_no in range(start_ln + 1, end_ln + 1):
                    interior_lines.add(ln_no)

    out: list[str] = []
    for idx, ln in enumerate(code_lines, start=1):
        if idx in interior_lines or not ln.strip():
            out.append(ln)
        else:
            out.append(f"{prefix}{ln}")
    return out


def replace_evolve_block(template_source: str, new_block_code: str) -> str:
    """Splice `new_block_code` into `template_source` between EVOLVE-BLOCK markers."""
    lines = template_source.splitlines()
    start_idx = -1
    end_idx = -1
    for idx, line in enumerate(lines):
        if line.strip() == EVOLVE_START_MARKER and line.startswith(EVOLVE_START_MARKER):
            start_idx = idx
        elif line.strip() == EVOLVE_END_MARKER and line.startswith(EVOLVE_END_MARKER):
            end_idx = idx
            break

    if start_idx < 0 or end_idx <= start_idx:
        raise SandboxViolationError("Canonical seed template is missing valid EVOLVE-BLOCK markers.")

    # Detect expected base indentation of the first code line inside template's EVOLVE-BLOCK
    template_inner = [ln for ln in lines[start_idx + 1 : end_idx] if ln.strip()]
    expected_indent = ""
    if template_inner:
        first_tmpl = template_inner[0]
        expected_indent = first_tmpl[: len(first_tmpl) - len(first_tmpl.lstrip(" "))]

    clean_inner = new_block_code.strip("\r\n")
    inner_lines = [
        ln
        for ln in clean_inner.splitlines()
        if ln.strip() not in (EVOLVE_START_MARKER, EVOLVE_END_MARKER)
    ]
    while inner_lines and not inner_lines[0].strip():
        inner_lines.pop(0)
    if not inner_lines:
        raise SandboxViolationError("Replacement EVOLVE-BLOCK has only blank lines.")

    first_cand = inner_lines[0]
    cand_indent = first_cand[: len(first_cand) - len(first_cand.lstrip(" "))]
    if expected_indent and not cand_indent:
        inner_lines = _indent_col0_block_preserving_multiline_strings(inner_lines, expected_indent)

    rebuilt = lines[: start_idx + 1] + inner_lines + lines[end_idx:]
    return "\n".join(rebuilt) + "\n"


def validate_ast_safety(
    source_code: str,
    *,
    required_functions: tuple[str, ...] = (),
    allowed_imports: frozenset[str] | None = None,
) -> ast.Module:
    """Parse Python source and reject forbidden imports, reflection, or missing entrypoints."""
    tree: ast.Module | None = None
    first_err: SyntaxError | None = None
    for candidate_src in (source_code, f"def _ae_ast_wrapper():\n{source_code}\n"):
        try:
            tree = ast.parse(candidate_src)
            break
        except SyntaxError as exc:
            if first_err is None:
                first_err = exc
    if tree is None:
        assert first_err is not None
        raise SandboxViolationError(
            f"Python syntax error at line {first_err.lineno}: {first_err.msg}"
        ) from first_err

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in FORBIDDEN_IMPORTS:
                    raise SandboxViolationError(
                        f"Forbidden module import '{alias.name}' blocked by AST sandbox."
                    )
                if allowed_imports is not None and root_pkg not in allowed_imports:
                    raise SandboxViolationError(
                        f"Disallowed module import '{alias.name}' (allowed: {sorted(allowed_imports)})."
                    )
        elif isinstance(node, ast.ImportFrom):
            root_pkg = (node.module or "").split(".")[0]
            if root_pkg in FORBIDDEN_IMPORTS:
                raise SandboxViolationError(
                    f"Forbidden from-import '{node.module}' blocked by AST sandbox."
                )
            if allowed_imports is not None and root_pkg and root_pkg not in allowed_imports:
                raise SandboxViolationError(
                    f"Disallowed from-import '{node.module}' (allowed: {sorted(allowed_imports)})."
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise SandboxViolationError(
                    f"Forbidden builtin call '{node.func.id}()' blocked by AST sandbox."
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CALLS:
                raise SandboxViolationError(
                    f"Forbidden attribute call '.{node.func.attr}()' blocked by AST sandbox."
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_DUNDER_ATTRS:
                raise SandboxViolationError(
                    f"Forbidden reflection attribute '{node.attr}' blocked by AST sandbox."
                )

    top_fn_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for req_fn in required_functions:
        if req_fn not in top_fn_names:
            raise SandboxViolationError(
                f"Required entrypoint function '{req_fn}' is missing from candidate program."
            )

    return tree


def validate_candidate_source(
    candidate_source: str,
    *,
    canonical_seed_source: str,
    required_functions: tuple[str, ...],
    allowed_imports: frozenset[str],
    require_markers: bool = True,
) -> tuple[str, str]:
    """Validate candidate EVOLVE-BLOCK + AST safety and splice into canonical immutable harness.

    Also checks that if a full candidate file with markers was supplied, no code outside
    EVOLVE-BLOCK was altered relative to `canonical_seed_source` (if altered, raises
    `SandboxViolationError` to block outer-harness reward hacking).

    Returns `(full_spliced_source, extracted_block)`.
    """
    block_code = extract_evolve_block(candidate_source, require_markers=require_markers)
    # Check AST safety of the candidate's own code first
    validate_ast_safety(candidate_source, allowed_imports=allowed_imports)

    # If candidate contains EVOLVE-BLOCK markers and non-empty outer lines, ensure it did not
    # tamper with outer helper constants/functions when attempting full-file submission.
    if EVOLVE_START_MARKER in candidate_source and EVOLVE_END_MARKER in candidate_source:
        cand_lines = candidate_source.splitlines()
        c_s = next(i for i, ln in enumerate(cand_lines) if ln.strip() == EVOLVE_START_MARKER)
        c_e = next(i for i, ln in enumerate(cand_lines) if ln.strip() == EVOLVE_END_MARKER)
        cand_prefix = "\n".join(ln.rstrip() for ln in cand_lines[:c_s]).strip()
        cand_suffix = "\n".join(ln.rstrip() for ln in cand_lines[c_e + 1 :]).strip()

        seed_lines = canonical_seed_source.splitlines()
        s_s = next(i for i, ln in enumerate(seed_lines) if ln.strip() == EVOLVE_START_MARKER)
        s_e = next(i for i, ln in enumerate(seed_lines) if ln.strip() == EVOLVE_END_MARKER)
        seed_prefix = "\n".join(ln.rstrip() for ln in seed_lines[:s_s]).strip()
        seed_suffix = "\n".join(ln.rstrip() for ln in seed_lines[s_e + 1 :]).strip()

        # Allow pure minimal `# EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END` snippets (empty prefix & suffix),
        # while rejecting full files that mutate immutable code outside the block!
        if (cand_prefix or cand_suffix) and (
            cand_prefix != seed_prefix or cand_suffix != seed_suffix
        ):
            raise SandboxViolationError(
                "Outer harness tampering detected: code outside EVOLVE-BLOCK markers must remain unmodified."
            )

    spliced_source = replace_evolve_block(canonical_seed_source, block_code)
    validate_ast_safety(
        spliced_source,
        required_functions=required_functions,
        allowed_imports=allowed_imports,
    )
    return spliced_source, block_code


def execute_candidate_module(
    spliced_source: str,
    module_name: str = "candidate_module",
) -> types.ModuleType:
    """Compile and execute validated candidate source in an isolated module namespace."""
    mod = types.ModuleType(module_name)
    compiled = compile(spliced_source, f"<{module_name}>", "exec")
    exec(compiled, mod.__dict__)  # noqa: S102 - AST-validated sandboxed source
    return mod


def run_with_timeout(fn: Callable[[], Any], timeout_sec: float = 15.0) -> Any:
    """Execute `fn` under a strict wall-clock timeout safe for both main and worker threads."""
    if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM"):
        def _handler(signum: int, frame: Any) -> None:
            raise CandidateTimeoutError(
                f"Candidate execution exceeded {timeout_sec:.1f}s timeout."
            )

        prev = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        try:
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, prev)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        raise CandidateTimeoutError(
            f"Candidate execution exceeded {timeout_sec:.1f}s timeout."
        ) from exc
    finally:
        executor.shutdown(wait=False)


def compute_unified_diff(
    seed_block: str,
    evolved_block: str,
    fromfile: str = "seed_evolve_block.py",
    tofile: str = "evolved_evolve_block.py",
) -> str:
    """Produce a clean unified diff between seed and evolved EVOLVE-BLOCK code."""
    diff_lines = difflib.unified_diff(
        seed_block.strip("\n").splitlines(keepends=True),
        evolved_block.strip("\n").splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile,
        n=3,
    )
    return "".join(diff_lines)
