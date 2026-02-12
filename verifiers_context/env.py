"""
ContextManagedRLMEnv — Learned context compression for multi-turn RLM conversations.

Extends RLMEnv to support multi-turn conversations where the model's context is
wiped between turns.  The only information that carries over is what the model
explicitly writes to ``answer["context"]``.  This forces the model to learn its
own compression strategy via RL, adhering to the Bitter Lesson: provide the
mechanism, let training discover the policy.

Multi-turn flow:
    Turn 1: Query Q1 → model uses REPL iteratively → sets answer["content"],
            answer["context"], answer["ready"] = True → MEMORY WIPE
    Turn 2: Model sees only: system prompt + answer["context"] from turn 1 + Q2
            → REPL iterations → answer → MEMORY WIPE
    ...
    Turn N: Final turn → answer["content"] is scored → reward propagates

The model discovers what to preserve in ``answer["context"]`` through the reward
signal alone.  No hand-engineered summarization, no fixed-window truncation.

This is a direct generalization of the Distributed-Inventory pattern
(https://ascl1u.github.io/blog/distributed-inventory/) to arbitrary multi-turn
RLM tasks.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import Messages, State
from verifiers.utils.message_utils import concat_messages

from .prompts import get_context_managed_system_prompt

logger = logging.getLogger(__name__)

# Code executed silently on the worker to reset between conversation turns.
# Preserves infrastructure (llm_batch, extra_data) but clears user state.
_WORKER_RESET_CODE = (
    '_rlm_ctx = answer.get("context", "") if isinstance(answer, dict) else ""\n'
    'answer = {"ready": False, "content": "", "context": ""}\n'
    "carried_context = _rlm_ctx\n"
    "del _rlm_ctx"
)

# Code to read answer["context"] from the worker after answer is ready.
_READ_CONTEXT_CODE = (
    "import json as _json\n"
    '_ctx = answer.get("context", "") if isinstance(answer, dict) else ""\n'
    "print(_json.dumps(_ctx))\n"
    "del _ctx"
)


class ContextManagedRLMEnv(RLMEnv):
    """RLM environment with learned context compression across conversation turns.

    Between conversation turns the model's full prompt context is wiped.
    The only carry-over channel is ``answer["context"]`` — a free-form string
    the model writes to before signaling ``answer["ready"] = True``.

    On the next turn the model sees:
    1. The system prompt (with RLM scaffolding)
    2. The value it wrote to ``answer["context"]`` on the previous turn
    3. The new query

    Everything else — REPL variables, previous tool outputs, previous model
    reasoning — is gone.  The model must learn through RL what information
    to compress and carry forward.

    Parameters
    ----------
    n_conversation_turns : int
        Number of user→RLM exchanges per episode (default: ``3``).
    context_max_length : int
        Hard truncation limit for ``answer["context"]`` in characters
        (default: ``8192``).  Not a soft budget — anything beyond is cut.
    queries_key : str
        Key in dataset ``info`` dict containing the list of per-turn queries
        (default: ``"queries"``).  Falls back to repeating the main prompt
        if the key is absent.
    **kwargs
        All standard ``RLMEnv`` parameters.
    """

    def __init__(
        self,
        *,
        n_conversation_turns: int = 3,
        context_max_length: int = 8192,
        queries_key: str = "queries",
        **kwargs: Any,
    ) -> None:
        if n_conversation_turns < 1:
            raise ValueError(
                f"n_conversation_turns must be >= 1, got {n_conversation_turns}"
            )
        # Set before super so setup_state can use them.
        self.n_conversation_turns = n_conversation_turns
        self.context_max_length = context_max_length
        self.queries_key = queries_key

        # Inject context-management instructions into the system prompt
        # unless the user already provided a custom one.
        if "system_prompt" not in kwargs:
            lang = kwargs.get("repl_language", "bash")
            verbosity = kwargs.get("root_prompt_verbosity", "light")
            kwargs["system_prompt"] = get_context_managed_system_prompt(
                lang, verbosity, n_conversation_turns
            )

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    async def setup_state(self, state: State, **kwargs: Any) -> State:
        state = await super().setup_state(state, **kwargs)

        # Extract per-turn queries from the dataset row.
        info = state.get("info") or {}
        if not isinstance(info, dict):
            info = {}

        queries = info.get(self.queries_key)
        if queries and isinstance(queries, list):
            # Pad or truncate to match n_conversation_turns.
            if len(queries) < self.n_conversation_turns:
                queries = list(queries) + [queries[-1]] * (
                    self.n_conversation_turns - len(queries)
                )
            else:
                queries = list(queries[: self.n_conversation_turns])
        else:
            # Fallback: repeat the main prompt for every turn.
            prompt = state.get("prompt", [])
            if isinstance(prompt, str):
                main_query = prompt
            elif isinstance(prompt, list) and prompt:
                last_user = next(
                    (
                        m.get("content", "")
                        for m in reversed(prompt)
                        if m.get("role") == "user"
                    ),
                    "",
                )
                main_query = last_user
            else:
                main_query = ""
            queries = [main_query] * self.n_conversation_turns

        state["_queries"] = queries
        state["_conversation_turn"] = 0
        state["_total_conversation_turns"] = self.n_conversation_turns
        state["_carried_context"] = ""
        state["_turn_answers"] = []
        state["_on_turn_boundary"] = False

        return state

    # ------------------------------------------------------------------
    # Core override: intercept answer-ready on intermediate turns
    # ------------------------------------------------------------------

    async def _call_repl(
        self,
        code: str,
        state: Any,
        *,
        ready_instruction: str,
        append_execution_time: bool,
    ) -> str:
        """Override: after super runs, intercept intermediate-turn answer-ready."""
        output = await super()._call_repl(
            code,
            state,
            ready_instruction=ready_instruction,
            append_execution_time=append_execution_time,
        )

        # Check if answer was marked ready on a non-final conversation turn.
        if "final_answer" in state and not self._is_final_conversation_turn(state):
            await self._transition_conversation_turn(state)

        return output

    async def _transition_conversation_turn(self, state: State) -> None:
        """Handle the transition between conversation turns.

        1. Read answer["context"] from the worker.
        2. Save the turn's answer.
        3. Reset the worker's answer dict for the next turn.
        4. Set the boundary flag so get_prompt_messages builds a fresh prompt.
        5. Remove final_answer so the rollout doesn't stop.
        """
        # 1. Read context from worker.
        ctx_result = await self._execute_code(_READ_CONTEXT_CODE, state)
        try:
            raw = (ctx_result.get("stdout") or "").strip()
            context = json.loads(raw) if raw else ""
        except (json.JSONDecodeError, TypeError):
            context = ""
        if not isinstance(context, str):
            context = str(context)

        # Truncate to budget.
        if len(context) > self.context_max_length:
            context = context[: self.context_max_length]

        # 2. Save turn results.
        state.setdefault("_turn_answers", []).append(state["final_answer"])
        state["_carried_context"] = context

        # 3. Reset worker for next turn.
        await self._execute_code(_WORKER_RESET_CODE, state)

        # 4. Set boundary flag and advance turn counter.
        state["_conversation_turn"] = state.get("_conversation_turn", 0) + 1
        state["_on_turn_boundary"] = True

        # 5. Remove final_answer so the rollout continues.
        del state["final_answer"]
        state.pop("final_env_response", None)

        logger.debug(
            "Conversation turn %d/%d — context length: %d chars",
            state["_conversation_turn"],
            state.get("_total_conversation_turns", 1),
            len(context),
        )

    # ------------------------------------------------------------------
    # Memory wipe: fresh prompt on turn boundaries
    # ------------------------------------------------------------------

    async def get_prompt_messages(self, state: State) -> Messages:
        """Override: build fresh prompt after a conversation turn boundary.

        On subsequent REPL turns within the same conversation turn, the
        normal RLMEnv prompt-building logic is used (concatenating trajectory).
        When a conversation turn boundary is crossed, we discard the full
        trajectory context and build a fresh prompt containing only:
        - RLM scaffolding + system prompt
        - The carried context from the previous turn
        - The new query
        """
        if len(state["trajectory"]) == 0:
            # Very first model call — use parent logic (injects scaffolding).
            return await super().get_prompt_messages(state)

        # For subsequent turns, parent calls env_response internally
        # (which triggers _call_repl → may set _on_turn_boundary).
        last_main = self._last_main_trajectory_step(state)
        if last_main is None:
            return await super().get_prompt_messages(state)

        prev_turn_prompt = last_main["prompt"]
        prev_turn_completion = last_main["completion"]
        messages = concat_messages([prev_turn_prompt, prev_turn_completion])

        # env_response processes tool calls — may trigger turn transition.
        env_resp = await self.env_response(messages, state)

        # Check if a conversation turn boundary was crossed.
        if state.get("_on_turn_boundary"):
            state["_on_turn_boundary"] = False
            return self._build_fresh_turn_prompt(state)

        # Normal case: concatenate as usual.
        return concat_messages([messages, env_resp])

    def _build_fresh_turn_prompt(self, state: State) -> Messages:
        """Build a clean prompt for a new conversation turn (memory wipe)."""
        turn = state.get("_conversation_turn", 0)
        queries = state.get("_queries", [])
        context = state.get("_carried_context", "")
        total = state.get("_total_conversation_turns", 1)

        query = queries[turn] if turn < len(queries) else ""

        system_prompt = state.get("rlm_system_prompt", "")
        scaffold = f"<RLM_SCAFFOLDING>\n{system_prompt}\n</RLM_SCAFFOLDING>\n\n"

        user_content = scaffold
        user_content += f"[Conversation Turn {turn + 1}/{total}]\n\n"

        if context:
            user_content += (
                "<CONTEXT_FROM_PREVIOUS_TURN>\n"
                f"{context}\n"
                "</CONTEXT_FROM_PREVIOUS_TURN>\n\n"
            )
        else:
            user_content += (
                "<CONTEXT_FROM_PREVIOUS_TURN>\n"
                "(empty — this is the first turn or no context was saved)\n"
                "</CONTEXT_FROM_PREVIOUS_TURN>\n\n"
            )

        user_content += query

        return cast(Messages, [{"role": "user", "content": user_content}])

    # ------------------------------------------------------------------
    # Stop condition overrides
    # ------------------------------------------------------------------

    @vf.stop
    async def no_tools_called(self, state: State) -> bool:
        """Override: don't stop during conversation turn transitions.

        After the model sets answer["ready"] on an intermediate turn, it may
        generate a text-only response (no tool calls).  Normally this would
        stop the rollout.  We suppress this during the transition so the
        fresh prompt for the next turn can be delivered.
        """
        if state.get("_on_turn_boundary"):
            return False
        return await super().no_tools_called(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_final_conversation_turn(self, state: State) -> bool:
        turn = state.get("_conversation_turn", 0)
        total = state.get("_total_conversation_turns", 1)
        return turn >= total - 1
