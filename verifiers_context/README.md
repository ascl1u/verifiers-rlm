# verifiers-context — Learned Context Compression for RLMs

An extension of [Prime Intellect's `verifiers` library](https://github.com/PrimeIntellect-ai/verifiers) that enables **learned context compression** over multi-turn RLM conversations.

## The Problem

The RLM looks like a normal LLM from the outside: text in, text out. But real-world use involves **multi-turn conversations** where context grows unboundedly. Current approaches to context management are hand-engineered: sliding windows, fixed summarization, RAG retrieval. These work, but they don't learn.

## The Approach — Bitter Lesson Compliance

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective." — Rich Sutton

Instead of engineering a compression strategy, we provide a **mechanism** and let the model discover its own **policy** through RL:

1. **Memory wipe**: Between conversation turns, the model's entire context window and REPL state are wiped.
2. **Single carry-over channel**: The only information that survives is what the model writes to `answer["context"]` — a free-form string.
3. **Sparse reward**: The model is scored only on its final answer. It must discover through training what information to compress and carry forward.

This directly generalizes the [Distributed-Inventory](https://ascl1u.github.io/blog/distributed-inventory/) pattern — an RL environment probing long-horizon reasoning through active context management — to arbitrary multi-turn RLM tasks.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Single RL Episode                    │
│                                                      │
│  Turn 1:  [System + Q1]                             │
│           → Model uses REPL iteratively             │
│           → Sets answer["content"], answer["context"]│
│           → answer["ready"] = True                  │
│           ═══ MEMORY WIPE ═══                       │
│                                                      │
│  Turn 2:  [System + carried_context + Q2]           │
│           → Model uses REPL (fresh state)           │
│           → Sets answer["content"], answer["context"]│
│           → answer["ready"] = True                  │
│           ═══ MEMORY WIPE ═══                       │
│                                                      │
│  Turn N:  [System + carried_context + QN]           │
│           → Model uses REPL (fresh state)           │
│           → Sets answer["content"] (SCORED)         │
│           → Reward propagates through full episode  │
└─────────────────────────────────────────────────────┘
```

## How It Works

`ContextManagedRLMEnv` extends `RLMEnv` with four surgical overrides:

| Override | Purpose |
|---|---|
| `_call_repl` | Intercepts `answer["ready"]` on intermediate turns — saves context, resets worker, prevents rollout from stopping |
| `get_prompt_messages` | After a turn boundary, builds a fresh prompt (system + carried context + new query) instead of concatenating trajectory history |
| `no_tools_called` | Suppresses the stop condition during turn transitions so the model gets the fresh prompt |
| `setup_state` | Initializes multi-turn tracking: per-turn queries, turn counter, carried context |

## Quick Start

```python
from verifiers_context import ContextManagedRLMEnv, from_query_lists
import verifiers as vf

# Build a multi-turn dataset
dataset = from_query_lists(
    queries=[
        ["Explore data.csv and summarize it.",
         "What is the mean of column 'price'?",
         "Filter rows where price > 100, report the count."],
    ],
    answers=["42"],
)

# Create the environment
env = ContextManagedRLMEnv(
    n_conversation_turns=3,      # 3 queries per episode
    context_max_length=4096,     # hard cap on carried context
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_turns=15,
)

# Reward only the final answer
@vf.reward
def score(completion, answer, **kwargs):
    return 1.0 if answer.strip() in completion.strip() else 0.0

# Train
trainer = vf.GRPOTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    env=env,
    train_dataset=dataset,
    reward_funcs=[score],
)
trainer.train()
```

## Dataset Helpers

```python
from verifiers_context import from_query_lists, make_multi_turn_dataset

# Option 1: Build directly from query sequences
ds = from_query_lists(
    queries=[["Q1", "Q2", "Q3"], ["Q4", "Q5", "Q6"]],
    answers=["A1", "A2"],
)

# Option 2: Convert single-turn dataset to multi-turn
from datasets import Dataset
single = Dataset.from_dict({"prompt": ["Q1","Q2","Q3","Q4"], "answer": ["A1","A2","A3","A4"]})
multi = make_multi_turn_dataset(single, n_turns=2)
# -> [["Q1","Q2"], ["Q3","Q4"]] with answers ["A2", "A4"]
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `n_conversation_turns` | `3` | Number of user→RLM exchanges per episode |
| `context_max_length` | `8192` | Hard truncation for `answer["context"]` (chars) |
| `queries_key` | `"queries"` | Dataset column containing per-turn query lists |
| `system_prompt` | auto | Override to provide custom system prompt |

All standard `RLMEnv` parameters (`model_name`, `max_turns`, `repl_language`, `root_prompt_verbosity`, etc.) are supported.

## Connection to Distributed-Inventory

This is a direct generalization of the [Distributed-Inventory](https://ascl1u.github.io/blog/distributed-inventory/) research, which demonstrated that LLMs can learn active context management through RL when forced to operate under strict memory constraints. The key insight: provide the mechanism (a carry-over channel), let training discover the policy (what to compress).
