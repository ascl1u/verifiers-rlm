"""
Example: Learned context compression across multi-turn data analysis.

Scenario — the model must answer a sequence of questions about a dataset,
but its context is wiped between turns.  It must learn to write useful
summaries into ``answer["context"]`` so that future turns have enough
information to answer correctly.

This is the simplest demonstration of ContextManagedRLMEnv.
"""

import verifiers as vf
from verifiers_context import ContextManagedRLMEnv, from_query_lists

# ── Dataset: each episode is a sequence of dependent questions ────────
# The model must carry forward enough state to answer later turns.

dataset = from_query_lists(
    queries=[
        [
            # Turn 1: Explore the data and build understanding.
            (
                "The file `sales.csv` contains columns: date, product, region, "
                "units, price. Read the file and report the total number of "
                "rows and the unique products."
            ),
            # Turn 2: Requires knowledge from turn 1.
            (
                "What is the total revenue (units * price) for each product? "
                "Report the product with the highest total revenue."
            ),
            # Turn 3: Builds on turn 2.
            (
                "For the top-revenue product you identified, which region "
                "had the highest units sold? Report the region name and count."
            ),
        ],
        [
            (
                "The file `employees.csv` has columns: id, name, department, "
                "salary, hire_date. Read and report the number of employees "
                "per department."
            ),
            (
                "What is the average salary per department? Which department "
                "has the highest average?"
            ),
            (
                "For the highest-average-salary department, who is the most "
                "recently hired employee? Report their name and hire date."
            ),
        ],
    ],
    answers=[
        "West, 450",     # Expected final answer for episode 1
        "Alice, 2024-11-15",  # Expected final answer for episode 2
    ],
)

# ── Environment ──────────────────────────────────────────────────────

env = ContextManagedRLMEnv(
    # Context compression settings
    n_conversation_turns=3,
    context_max_length=4096,
    queries_key="queries",

    # Standard RLM settings
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_turns=15,           # REPL turns per conversation turn
    root_prompt_verbosity="heavy",
)


# ── Reward: score only the final turn's answer ───────────────────────
# The model only gets a reward signal at the end.  This forces it to
# learn what to compress and carry forward.

@vf.reward
def final_answer_reward(completion, answer, **kwargs):
    """Simple exact-match reward on the last turn's answer."""
    if not completion or not answer:
        return 0.0
    return 1.0 if answer.strip().lower() in completion.strip().lower() else 0.0


# ── Training (GRPO) ─────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = vf.GRPOTrainer(
        model="Qwen/Qwen2.5-7B-Instruct",
        env=env,
        train_dataset=dataset,
        reward_funcs=[final_answer_reward],
        run_name="context-compression-demo",
        per_device_train_batch_size=2,
        num_generations=4,
    )
    trainer.train()
