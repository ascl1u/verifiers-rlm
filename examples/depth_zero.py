"""
Depth=0 example: LLM + REPL, no sub-LLM capability.

Useful for tasks where code execution is sufficient and sub-LLM overhead
is undesirable.  The model has a full Python REPL but cannot call
llm_batch().  No interception server is started.

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
        recursion_depth=0,
        dataset=dataset,
        rubric=rubric,
        repl_language="python",
        root_prompt_verbosity="medium",
        max_iterations=10,
    )
    return env
