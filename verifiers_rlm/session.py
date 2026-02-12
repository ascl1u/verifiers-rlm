"""
In-process REPL session for nested sub-LLMs.

Provides lightweight Python code execution via exec() with isolated namespaces,
mirroring the behavior of the external worker script but running entirely
in-process for zero subprocess overhead.

This is the key architectural choice that makes depth > 1 practical: instead
of spinning up a full worker process for each nested sub-LLM, we execute code
in-process with namespace isolation.
"""

from __future__ import annotations

import ast
import contextlib
import io
import traceback
from typing import Any


class InProcessReplSession:
    """Lightweight in-process Python REPL for nested sub-LLM code execution.

    Provides an isolated namespace with an ``answer`` variable for the sub-LLM
    to write results into. Code execution mirrors the external worker's behavior:

    - Sequential statements are executed via ``exec``
    - A trailing expression's value is captured via ``eval``
    - stdout/stderr are captured
    - Errors return full tracebacks

    Thread Safety:
        ``execute_sync`` is designed to run in a ``ThreadPoolExecutor`` (single
        thread) so that synchronous code can call ``llm_batch()`` which bridges
        to the async event loop via ``run_coroutine_threadsafe``. The namespace
        is only accessed from that single worker thread during execution.
    """

    def __init__(self) -> None:
        self.namespace: dict[str, Any] = {
            "__name__": "__main__",
            "answer": {"ready": False, "content": ""},
        }
        self.execution_count: int = 0

    def execute_sync(self, code: str) -> dict[str, Any]:
        """Execute Python code synchronously. Intended for thread pool execution.

        Mirrors the external worker script's execution model:
        - Parses code as a module
        - Executes all statements except the last if it's a bare expression
        - Evaluates the trailing expression and captures its repr
        - Captures stdout, stderr, and exceptions

        Returns:
            Dict with keys: ``status``, ``stdout``, ``stderr``, ``result``.
        """
        self.execution_count += 1
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        result: dict[str, Any] = {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "result": None,
        }

        try:
            with (
                contextlib.redirect_stdout(stdout_buffer),
                contextlib.redirect_stderr(stderr_buffer),
            ):
                module_ast = ast.parse(code, mode="exec")
                body = list(module_ast.body)
                trailing_expr = None
                if body and isinstance(body[-1], ast.Expr):
                    trailing_expr = body.pop()
                if body:
                    exec_module = ast.Module(body=body, type_ignores=[])
                    exec(
                        compile(exec_module, "<repl>", "exec"),
                        self.namespace,
                        self.namespace,
                    )
                if trailing_expr is not None:
                    value = eval(
                        compile(
                            ast.Expression(trailing_expr.value), "<repl>", "eval"
                        ),
                        self.namespace,
                        self.namespace,
                    )
                    if value is not None:
                        result["result"] = repr(value)
        except Exception:
            result["status"] = "error"
            result["result"] = traceback.format_exc()

        result["stdout"] = stdout_buffer.getvalue()
        result["stderr"] = stderr_buffer.getvalue()
        return result

    @property
    def answer_ready(self) -> bool:
        """Whether the sub-LLM has signaled its answer is complete."""
        answer = self.namespace.get("answer", {})
        if isinstance(answer, dict):
            return bool(answer.get("ready", False))
        return False

    @property
    def answer_content(self) -> str:
        """The sub-LLM's current answer content."""
        answer = self.namespace.get("answer", {})
        if isinstance(answer, dict):
            return str(answer.get("content", ""))
        return ""
