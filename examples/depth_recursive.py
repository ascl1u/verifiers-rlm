"""
Depth=3 example: Hierarchical decomposition with nested sub-LLMs.

The root model can spawn sub-LLMs (depth 1), which themselves can spawn
deeper sub-LLMs (depth 2), each with their own Python REPL.  At depth 3
(max), sub-LLMs are flat (no REPL, no llm_batch).

This enables hierarchical problem decomposition:
  - Root (depth 0): Orchestrates overall strategy via bash/python REPL
  - Depth 1 sub-LLMs: Handle major sub-tasks with in-process Python REPL
  - Depth 2 sub-LLMs: Handle focused sub-sub-tasks with REPL
  - Depth 3 sub-LLMs: Simple text-in/text-out responses

Usage:
    prime eval run . -m gpt-5-mini
"""

import verifiers as vf

from verifiers_rlm import RecursiveRLMEnv


def load_environment() -> vf.Environment:
    dataset = vf.load_example_dataset("gsm8k")

    async def correct_answer(completion, answer) -> float:
        completion_ans = completion[-1]["content"]
        return 1.0 if completion_ans == answer else 0.0

    rubric = vf.Rubric(funcs=[correct_answer])

    env = RecursiveRLMEnv(
        recursion_depth=3,
        nested_max_iterations=5,
        nested_max_output_length=4096,
        dataset=dataset,
        rubric=rubric,
        repl_language="python",
        root_prompt_verbosity="medium",
        max_iterations=20,
    )
    return env
