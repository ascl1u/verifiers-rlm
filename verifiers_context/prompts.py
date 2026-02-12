"""
System prompts for ContextManagedRLMEnv.

Each prompt extends the standard RLM prompt with context-management
instructions: the ``answer["context"]`` carry-over mechanism, the memory
wipe between turns, and ``carried_context`` availability in the REPL.
"""

from __future__ import annotations

# ── Context management addendum ──────────────────────────────────────
# Appended to every system prompt.  Kept concise to avoid eating context
# budget — the model should learn the policy, not memorize the manual.

_CONTEXT_ADDENDUM_PYTHON = """
## Context Management — Multi-Turn Conversation

This environment has **{n_turns} conversation turns**.  Between turns your
full context window and REPL state are **wiped**.

The **only** information that survives the wipe is what you write to
`answer["context"]`.  On the next turn it appears as the variable
`carried_context` in your REPL.

### How it works

1. Process the current query using the REPL.
2. Before finishing, write anything you need for future turns to
   `answer["context"]` (a free-form string, max {ctx_limit} chars).
3. Set `answer["content"]` to your response and `answer["ready"] = True`.
4. On the next turn you start fresh with only `carried_context` and
   the new query.

Think carefully about *what* to preserve.
"""

_CONTEXT_ADDENDUM_BASH = """
## Context Management — Multi-Turn Conversation

This environment has **{n_turns} conversation turns**.  Between turns your
full context window and shell state are **wiped**.

The **only** information that survives the wipe is what you write to
`RLM_CONTEXT` before calling `export RLM_READY=1`.  On the next turn it
appears as the environment variable `CARRIED_CONTEXT`.

### How it works

1. Process the current query using the shell.
2. Before finishing, write anything you need for future turns to
   `RLM_CONTEXT` (a free-form string).
3. Set `RLM_CONTENT` to your response and `export RLM_READY=1`.
4. On the next turn you start fresh with only `CARRIED_CONTEXT` and
   the new query.

Think carefully about *what* to preserve.
"""


# ── Base prompts (mirroring RLMEnv's store) ──────────────────────────

_PYTHON_BASE = {
    "light": (
        'You have the `call_python_repl` tool and a filesystem available to you.\n'
        '\n'
        'There exists an `answer` variable, which is a dict. '
        '`answer["content"]` must contain your answer. '
        'When the final answer is set, set `answer["ready"] = True`.\n'
    ),
    "medium": (
        'You have the `call_python_repl` tool and a filesystem available to you.\n'
        '\n'
        'There exists an `answer` variable, which is a dict. '
        '`answer["content"]` must contain your answer. '
        'When the final answer is set, set `answer["ready"] = True`.\n'
        '\n'
        'This is an iterative environment. '
        'Make use of sub-LLMs via `llm_batch` whenever they could be useful; '
        'prefer calling them in parallel to calling them sequentially.\n'
    ),
    "heavy": (
        'You are operating in a Recursive Language Model (RLM) environment - '
        'an iterative Python REPL where you explore data step by step.\n'
        '\n'
        'A filesystem is available; explore it as needed.\n'
        '\n'
        '## Critical: This is an ITERATIVE environment\n'
        '\n'
        'You will write code, see its output, then write more code based on '
        'what you learned. **Do NOT try to solve everything in one tool call.** '
        'Each tool call executes and returns output before you continue.\n'
        '\n'
        'Use the `call_python_repl` tool to execute Python code. The REPL '
        'maintains state across calls. See the tool description for available '
        'variables and functions.\n'
        '\n'
        '## Workflow\n'
        '\n'
        '**Step 1: Explore the filesystem**\n'
        '```python\n'
        'import os\n'
        'print(os.getcwd())\n'
        'print(os.listdir("."))\n'
        '```\n'
        'Wait for output. Now you know the actual format.\n'
        '\n'
        '**Step 2: Process and build your answer**\n'
        '```python\n'
        'answer["content"] = "your current best answer"\n'
        '```\n'
        '\n'
        '**Step 3: Verify and finalize (only after reviewing output)**\n'
        '```python\n'
        'print(f"My answer: {answer[\'content\']}")\n'
        'answer["ready"] = True\n'
        '```\n'
        '\n'
        '## Important Rules\n'
        '\n'
        '1. **NEVER set `answer["ready"] = True` until you have seen '
        'execution output** - you need feedback first\n'
        '2. **One step at a time** - make small tool calls, see output, '
        'then continue\n'
        '3. **Use `llm_batch()` for semantic tasks** - summarization, '
        'understanding text, classification, etc.\n'
        '   Pass a list of strings only (no message dicts).\n'
    ),
}

_BASH_BASE = {
    "light": (
        'You have the `call_bash_repl` tool and a filesystem available to you.\n'
        '\n'
        'In the end, the `RLM_CONTENT` environment variable must contain your '
        'answer. When the final answer is set, call `export RLM_READY=1`.\n'
    ),
    "medium": (
        'You have the `call_bash_repl` tool and a filesystem available to you.\n'
        '\n'
        'In the end, the `RLM_CONTENT` environment variable must contain your '
        'answer. When the final answer is set, call `export RLM_READY=1`.\n'
        '\n'
        'This is an iterative environment. '
        'Make use of sub-LLMs via `llm_batch` whenever they could be useful; '
        'prefer calling them in parallel to calling them sequentially.\n'
    ),
    "heavy": (
        'You are operating in a Recursive Language Model (RLM) environment - '
        'an iterative Bash REPL where you explore data step by step.\n'
        '\n'
        'A filesystem is available; explore it as needed.\n'
        '\n'
        '## Critical: This is an ITERATIVE environment\n'
        '\n'
        'You will run shell commands, see their output, then run more commands '
        'based on what you learned. **Do NOT try to solve everything in one '
        'tool call.** Each tool call executes and returns output before you '
        'continue.\n'
        '\n'
        'Use the `call_bash_repl` tool to execute Bash commands. The shell '
        'maintains state across calls. See the tool description for available '
        'variables and commands.\n'
        '\n'
        '## Workflow\n'
        '\n'
        '**Step 1: Explore the filesystem**\n'
        '```bash\n'
        'pwd\n'
        'ls\n'
        '```\n'
        'Wait for output. Now you know the actual format.\n'
        '\n'
        '**Step 2: Build your answer**\n'
        '```bash\n'
        'export RLM_CONTENT="your current best answer"\n'
        '```\n'
        '\n'
        '**Step 3: Verify and finalize (only after reviewing output)**\n'
        '```bash\n'
        'printf "My answer: %s\\n" "$RLM_CONTENT"\n'
        'export RLM_READY=1\n'
        '```\n'
        '\n'
        '## Important Rules\n'
        '\n'
        '1. **NEVER set `RLM_READY=1` until you have seen execution output** '
        '- you need feedback first\n'
        '2. **One step at a time** - make small tool calls, see output, '
        'then continue\n'
        '3. **Use `llm_batch` for semantic tasks** - summarization, '
        'understanding text, classification, etc.\n'
        '   Pass a list of strings only (no message dicts).\n'
    ),
}


# ── Public API ───────────────────────────────────────────────────────

DEFAULT_CONTEXT_MAX_LENGTH = 8192


def get_context_managed_system_prompt(
    language: str = "python",
    verbosity: str = "light",
    n_conversation_turns: int = 3,
    context_max_length: int = DEFAULT_CONTEXT_MAX_LENGTH,
) -> str:
    """Build a system prompt with context-management instructions.

    Parameters
    ----------
    language : str
        REPL language: ``"python"`` or ``"bash"``.
    verbosity : str
        Prompt verbosity level: ``"light"``, ``"medium"``, or ``"heavy"``.
    n_conversation_turns : int
        Number of conversation turns per episode.
    context_max_length : int
        Maximum characters for ``answer["context"]``.

    Returns
    -------
    str
        Composite system prompt.
    """
    if language == "bash":
        base = _BASH_BASE.get(verbosity, _BASH_BASE["light"])
        addendum = _CONTEXT_ADDENDUM_BASH
    else:
        base = _PYTHON_BASE.get(verbosity, _PYTHON_BASE["light"])
        addendum = _CONTEXT_ADDENDUM_PYTHON

    return base + addendum.format(
        n_turns=n_conversation_turns,
        ctx_limit=f"{context_max_length:,}",
    )
