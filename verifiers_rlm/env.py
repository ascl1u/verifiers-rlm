"""
RecursiveRLMEnv — Variable recursion depth for the Recursive Language Model.

Extends verifiers' RLMEnv with a ``recursion_depth`` parameter:

- **depth=0**: LLM + REPL + user tools (no sub-LLM calls).
  No interception server, no ``llm_batch``.  Useful when code execution
  alone suffices and sub-LLM overhead is undesirable.

- **depth=1**: Standard RLM behavior (backward compatible with ``RLMEnv``).
  Root LLM + REPL + flat sub-LLMs via ``llm_batch``.

- **depth=N** (N > 1): Sub-LLMs at each level below the maximum get their own
  lightweight in-process Python REPL and can spawn deeper sub-LLMs via
  ``llm_batch()``.  At depth=N, sub-LLMs are flat (text-in, text-out with
  optional tool calling).

Architecture for depth > 1:
    Instead of spinning up external workers for each nested sub-LLM (expensive),
    we use ``InProcessReplSession``—code executes via ``exec()`` in an isolated
    namespace within the framework process.  ``llm_batch()`` is injected as a
    sync function that bridges to the async event loop via
    ``run_coroutine_threadsafe``, with code execution running in a
    ``ThreadPoolExecutor`` to avoid blocking the loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, cast

from verifiers.envs.experimental.rlm_env import (
    RLMEnv,
    SubLLMResult,
    SubLLMTurn,
    _extract_tokens_from_response,
)
from verifiers.types import ChatMessage, ChatMessages, State

from .prompts import get_depth_zero_system_prompt, get_nested_system_prompt
from .session import InProcessReplSession

logger = logging.getLogger(__name__)


class RecursiveRLMEnv(RLMEnv):
    """Recursive Language Model environment with configurable recursion depth.

    Extends ``RLMEnv`` with a single ``recursion_depth`` parameter that unlocks
    three operating modes:

    - ``recursion_depth=0``: Stripped-down mode.  LLM + REPL + user tools only.
      No ``llm_batch``, no interception server overhead.
    - ``recursion_depth=1``: Standard RLM (backward compatible with ``RLMEnv``).
    - ``recursion_depth=N`` (N > 1): Sub-LLMs at depth ``d < N`` get their own
      in-process Python REPL and access to ``llm_batch()`` for spawning deeper
      sub-LLMs.  At ``depth=N``, sub-LLMs are flat.

    Parameters
    ----------
    recursion_depth : int
        Maximum recursion depth for sub-LLM calls (default: ``1``).
    nested_max_iterations : int
        Maximum REPL iterations for nested sub-LLMs (default: ``10``).
    nested_max_output_length : int
        Maximum character length of code execution output shown to nested
        sub-LLMs (default: ``4096``).
    **kwargs
        All standard ``RLMEnv`` parameters (``dataset``, ``rubric``,
        ``repl_language``, ``max_iterations``, ``tools``, ``sub_tools``, etc.).
    """

    def __init__(
        self,
        *,
        recursion_depth: int = 1,
        nested_max_iterations: int = 10,
        nested_max_output_length: int = 4096,
        **kwargs: Any,
    ) -> None:
        if recursion_depth < 0:
            raise ValueError(
                f"recursion_depth must be >= 0, got {recursion_depth}"
            )

        # Set before super().__init__ so _build_fixed_root_tools sees it.
        self.recursion_depth = recursion_depth
        self.nested_max_iterations = nested_max_iterations
        self.nested_max_output_length = nested_max_output_length

        # For depth=0, provide a system prompt that doesn't mention llm_batch.
        if recursion_depth == 0 and "system_prompt" not in kwargs:
            lang = kwargs.get("repl_language", "bash")
            verbosity = kwargs.get("root_prompt_verbosity", "light")
            kwargs["system_prompt"] = get_depth_zero_system_prompt(lang, verbosity)

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Depth=0: disable sub-LLM infrastructure
    # ------------------------------------------------------------------

    def _build_fixed_root_tools(self) -> list[Callable]:
        """Override: return empty list at depth=0 (no ``llm_batch``)."""
        if self.recursion_depth == 0:
            return []
        return super()._build_fixed_root_tools()

    async def _setup_interception_and_register(
        self, state: State, rollout_id: str
    ) -> State:
        """Override: skip interception server when no sub-LLM calls are possible."""
        if self.recursion_depth == 0:
            # No sub-LLM calls will be made — stub out the URLs.
            state["interception_url"] = ""
            state["root_tool_url"] = ""
            return state
        return await super()._setup_interception_and_register(state, rollout_id)

    # ------------------------------------------------------------------
    # Depth > 1: recursive sub-LLM execution
    # ------------------------------------------------------------------

    async def _run_sub_llm(
        self, state: State, client: Any, model: str, messages: ChatMessages
    ) -> SubLLMResult:
        """Override: route depth-1 sub-LLM calls based on recursion_depth.

        This is the entry point called from the HTTP interception path
        (root REPL → ``llm_batch`` → interception server → here).
        At depth=1, if ``recursion_depth > 1``, the sub-LLM gets its own REPL.
        """
        if self.recursion_depth > 1:
            return await self._run_recursive_sub_llm(
                state, client, model, messages, depth=1
            )
        # Standard behavior: flat sub-LLM with optional tool-calling loop.
        return await super()._run_sub_llm(state, client, model, messages)

    async def _run_sub_llm_at_depth(
        self,
        state: State,
        client: Any,
        model: str,
        messages: ChatMessages,
        depth: int,
    ) -> SubLLMResult:
        """Run a sub-LLM at an arbitrary depth, routing to REPL or flat mode.

        Called by nested ``llm_batch()`` to dispatch sub-sub-LLM calls.
        """
        if depth >= self.recursion_depth:
            # At max depth: fall through to the base flat implementation.
            return await RLMEnv._run_sub_llm(self, state, client, model, messages)
        return await self._run_recursive_sub_llm(
            state, client, model, messages, depth
        )

    async def _run_recursive_sub_llm(
        self,
        state: State,
        client: Any,
        model: str,
        messages: ChatMessages,
        depth: int,
    ) -> SubLLMResult:
        """Run a sub-LLM with its own in-process REPL at the given depth.

        The sub-LLM receives:
        - ``call_python_repl``: in-process code execution via isolated namespace
        - User-defined sub-tools (same as flat sub-LLMs)
        - ``llm_batch`` (in REPL namespace): available if ``depth + 1 < max``

        Code execution runs in a ``ThreadPoolExecutor`` so that synchronous
        ``llm_batch()`` calls can bridge to the async event loop via
        ``run_coroutine_threadsafe``.
        """
        session = InProcessReplSession()
        loop = asyncio.get_running_loop()
        thread_pool = ThreadPoolExecutor(max_workers=1)

        # Inject llm_batch into the REPL namespace if not at the last
        # recursion level before flat.
        has_llm_batch = (depth + 1) < self.recursion_depth
        if has_llm_batch:
            self._inject_nested_llm_batch(
                session, state, client, model, depth, loop
            )

        # Build OAI tool definitions for this nested sub-LLM.
        oai_tools = self._build_nested_oai_tools(has_llm_batch=has_llm_batch)

        # Multi-turn REPL loop.
        current_messages = list(messages)
        turns: list[SubLLMTurn] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        tool_call_count = 0

        try:
            for _iteration in range(self.nested_max_iterations):
                # --- LLM call ---
                response = await self._call_sub_llm_api(
                    state, client, model, current_messages, oai_tools
                )
                if response is None:
                    # Timed out.
                    break

                pt, ct = _extract_tokens_from_response(response)
                total_prompt_tokens += pt
                total_completion_tokens += ct

                assistant_msg = response.choices[0].message
                tcs = getattr(assistant_msg, "tool_calls", None)

                turns.append(
                    SubLLMTurn(
                        prompt_messages=[
                            cast(ChatMessage, dict(m)) for m in current_messages
                        ],
                        response=response,
                        tool_call_count=len(tcs) if tcs else 0,
                    )
                )

                # No tool calls → model has finished.
                if not tcs:
                    break

                current_messages.append(
                    cast(ChatMessage, assistant_msg.model_dump())
                )

                # --- Process each tool call ---
                for tc in tcs:
                    tool_call_count += 1
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    if name == "call_python_repl":
                        code = args.get("code", "")
                        try:
                            exec_result = await asyncio.wait_for(
                                loop.run_in_executor(
                                    thread_pool, session.execute_sync, code
                                ),
                                timeout=self.code_execution_timeout,
                            )
                        except asyncio.TimeoutError:
                            exec_result = {
                                "status": "error",
                                "stdout": "",
                                "stderr": "",
                                "result": (
                                    f"Code execution timed out after "
                                    f"{self.code_execution_timeout}s"
                                ),
                            }
                        result_content = self._format_nested_exec_output(
                            exec_result
                        )
                    elif name in self.sub_tool_map:
                        # User-defined sub-tool — _call_sub_tool returns a
                        # full tool message dict.
                        tool_msg = await self._call_sub_tool(
                            name, args, tc.id
                        )
                        current_messages.append(cast(ChatMessage, tool_msg))
                        continue
                    else:
                        result_content = f"Error: Unknown tool '{name}'"

                    current_messages.append(
                        cast(
                            ChatMessage,
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_content,
                            },
                        )
                    )

                # Check if the sub-LLM signaled completion.
                if session.answer_ready:
                    break
        finally:
            thread_pool.shutdown(wait=False)

        # Determine final content.
        if session.answer_ready:
            final_content = session.answer_content
        elif turns:
            last_msg = turns[-1]["response"].choices[0].message
            final_content = getattr(last_msg, "content", None) or ""
        else:
            final_content = ""

        return SubLLMResult(
            final_content=final_content,
            turns=turns,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            tool_call_count=tool_call_count,
            num_turns=len(turns),
            max_turns_reached=len(turns) >= self.nested_max_iterations,
        )

    # ------------------------------------------------------------------
    # Helpers for nested sub-LLM execution
    # ------------------------------------------------------------------

    def _inject_nested_llm_batch(
        self,
        session: InProcessReplSession,
        state: State,
        client: Any,
        model: str,
        depth: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Inject ``llm_batch`` as a sync function into the REPL namespace.

        Uses ``run_coroutine_threadsafe`` to bridge sync ``exec()`` context
        to async sub-LLM API calls.  The calling code must be running in a
        ``ThreadPoolExecutor`` thread (not the event loop thread).
        """
        env_ref = self  # Captured by closure.

        async def _batch_async(prompts: list[str]) -> list[str]:
            """Dispatch a batch of prompts to sub-LLMs at ``depth + 1``."""
            semaphore = asyncio.Semaphore(env_ref.max_sub_llm_parallelism)
            results: list[str | None] = [None] * len(prompts)

            async def _call_one(index: int, prompt: str) -> None:
                async with semaphore:
                    system_prompt = get_nested_system_prompt(
                        depth + 1,
                        env_ref.recursion_depth,
                        env_ref.nested_max_iterations,
                    )
                    msgs: ChatMessages = [
                        cast(
                            ChatMessage,
                            {"role": "system", "content": system_prompt},
                        ),
                        cast(
                            ChatMessage,
                            {"role": "user", "content": prompt},
                        ),
                    ]
                    try:
                        result = await env_ref._run_sub_llm_at_depth(
                            state, client, model, msgs, depth + 1
                        )
                        results[index] = result["final_content"]
                    except Exception as exc:
                        logger.warning(
                            "Nested llm_batch[%d] at depth %d failed: %s",
                            index,
                            depth + 1,
                            exc,
                        )
                        results[index] = f"Error in sub-LLM call: {exc}"

            await asyncio.gather(
                *[_call_one(i, p) for i, p in enumerate(prompts)]
            )
            return [r or "" for r in results]

        def llm_batch(prompts: list[str]) -> list[str]:
            """Call sub-LLMs on multiple prompts in parallel.

            This is a **synchronous** wrapper intended to be called from within
            ``exec()`` running in a thread pool.  It dispatches the async batch
            to the event loop and blocks until complete.

            Args:
                prompts: List of prompt strings.

            Returns:
                List of response strings in the same order as the prompts.
            """
            if not isinstance(prompts, list):
                raise ValueError("llm_batch expects a list of prompt strings")
            # Generous timeout: scale with batch size.
            timeout = env_ref.sub_llm_api_timeout * max(1, len(prompts))
            future = asyncio.run_coroutine_threadsafe(
                _batch_async(prompts), loop
            )
            return future.result(timeout=timeout)

        session.namespace["llm_batch"] = llm_batch

    def _build_nested_oai_tools(
        self, *, has_llm_batch: bool = False
    ) -> list[dict[str, Any]]:
        """Build OAI tool definitions for a nested sub-LLM.

        Always includes ``call_python_repl``.  Includes user-defined sub-tools
        if any are configured.  ``llm_batch`` is *not* an OAI tool — it's a
        Python function injected into the REPL namespace (consistent with the
        root-level design).
        """
        llm_batch_note = ""
        if has_llm_batch:
            llm_batch_note = (
                " `llm_batch(prompts)` is available in the namespace "
                "to delegate sub-tasks to further sub-LLMs."
            )

        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "call_python_repl",
                    "description": (
                        "Execute Python code in a persistent REPL session. "
                        "Variables persist across calls. An `answer` dict is "
                        "available: set answer['content'] to your result and "
                        "answer['ready'] = True when done."
                        + llm_batch_note
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute",
                            }
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

        # Include user-defined sub-tools.
        if self.sub_oai_tools:
            tools.extend(self.sub_oai_tools)

        return tools

    def _format_nested_exec_output(self, result: dict[str, Any]) -> str:
        """Format in-process REPL execution result for the nested sub-LLM."""
        parts: list[str] = []

        stdout = result.get("stdout", "")
        if stdout:
            if len(stdout) > self.nested_max_output_length:
                stdout = (
                    stdout[: self.nested_max_output_length]
                    + "\n... [output truncated]"
                )
            parts.append(stdout)

        stderr = result.get("stderr", "")
        if stderr:
            parts.append(f"[stderr] {stderr}")

        expr_result = result.get("result")
        if expr_result is not None:
            if result.get("status") == "error":
                parts.append(f"[error]\n{expr_result}")
            else:
                parts.append(expr_result)

        return "\n".join(parts) if parts else "[No output]"
