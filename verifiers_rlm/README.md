# verifiers-rlm

**Variable recursion depth for the [Recursive Language Model](https://www.primeintellect.ai/blog/rlm) environment in [verifiers](https://github.com/PrimeIntellect-ai/verifiers).**

Implements configurable recursion depth, enabling the RLM to operate at depth 0 (no sub-LLMs), depth 1 (standard), or arbitrary depth N (hierarchical sub-LLM decomposition).

## Motivation

The current `RLMEnv` operates at a fixed recursion depth of 1: the root model has a Python REPL and can spawn flat sub-LLMs via `llm_batch()`, but sub-LLMs cannot recurse further. This limits the RLM's ability to decompose problems hierarchically.

> Right now, the RLM has a recursion depth of exactly 1. We plan on making it possible to decrease that recursion depth to 0 (so that we have a normal LLM with a Python REPL containing the input data, and access to all user-added tools), and to increase it arbitrarily, so that sub-LLMs can call further sub-LLMs.

`verifiers-rlm` extends `RLMEnv` with a single `recursion_depth` parameter:

| Depth | Root Model | Sub-LLMs (depth 1..N-1) | Leaf Sub-LLMs (depth N) |
|-------|-----------|------------------------|------------------------|
| **0** | REPL + tools | — | — |
| **1** | REPL + tools + `llm_batch` | — | Flat (text in, text out) |
| **N** | REPL + tools + `llm_batch` | In-process REPL + `llm_batch` | Flat (text in, text out) |


## Installation

```bash
uv add verifiers-rlm
```

## Usage

### Depth 0: REPL-Only Mode

```python
from verifiers_rlm import RecursiveRLMEnv

env = RecursiveRLMEnv(
    recursion_depth=0,
    dataset=dataset,
    rubric=rubric,
    repl_language="python",
)
```

The model gets `call_python_repl` + user tools. No `llm_batch`, no interception server overhead.

### Depth 1: Standard RLM (Backward Compatible)

```python
env = RecursiveRLMEnv(
    recursion_depth=1,  # default
    dataset=dataset,
    rubric=rubric,
)
```

Identical to the standard `RLMEnv`. Root gets REPL + `llm_batch`, sub-LLMs are flat.

### Depth N: Hierarchical Recursion

```python
env = RecursiveRLMEnv(
    recursion_depth=3,
    nested_max_iterations=5,
    nested_max_output_length=4096,
    dataset=dataset,
    rubric=rubric,
    repl_language="python",
)
```

Sub-LLMs at depths 1 and 2 get their own Python REPL with `llm_batch()` in the namespace. At depth 3, sub-LLMs are flat.

## Architecture

### Nested Sub-LLM Execution

For `recursion_depth > 1`, sub-LLMs below the maximum depth receive:

1. **In-process REPL** (`InProcessReplSession`): Code executes via `exec()` in an isolated namespace, mirroring the external worker's behavior but with zero subprocess overhead.

2. **`llm_batch()` in namespace**: A sync function bridged to the async event loop via `run_coroutine_threadsafe`. Code runs in a `ThreadPoolExecutor` so the sync `llm_batch()` can block while the event loop processes the async sub-LLM API calls.

3. **User sub-tools**: The same tools available to flat sub-LLMs.

```
Root Model (depth 0)
├── call_python_repl → External Worker (subprocess)
│   └── llm_batch() → HTTP → Interception Server
│       └── Sub-LLM (depth 1)
│           ├── call_python_repl → InProcessReplSession (exec)
│           │   └── llm_batch() → run_coroutine_threadsafe
│           │       └── Sub-LLM (depth 2)
│           │           ├── call_python_repl → InProcessReplSession
│           │           └── (no llm_batch at max depth)
│           └── sub-tools
└── sub-tools
```

### Design Decisions

- **In-process execution for nested REPLs**: Spinning up external workers for each sub-LLM would be prohibitively expensive. In-process `exec()` with isolated namespaces provides the same capability with minimal overhead. The trade-off is weaker sandboxing, which is acceptable because nested sub-LLMs execute model-generated code in the same trust domain as the root worker.

- **Thread-based sync-async bridge**: The `exec()` → `llm_batch()` → async LLM call chain requires bridging sync and async contexts. Running code in a `ThreadPoolExecutor` allows the sync `llm_batch` to block its thread while `run_coroutine_threadsafe` dispatches to the event loop. This is the standard pattern for this problem and avoids nested event loops.

- **Python-only nested REPLs**: While the root model supports both bash and python REPLs, nested sub-LLMs always use Python. In-process bash execution would require subprocess management and PTY handling, negating the lightweight design. Python's `exec()` is the natural fit for the in-process bridge architecture.

- **Depth-aware system prompts**: Each nested sub-LLM receives a system prompt that matches its capabilities. Sub-LLMs with `llm_batch` are told about it; those at max depth are not. This prevents the model from attempting to call unavailable functions.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recursion_depth` | `1` | Maximum sub-LLM nesting depth (0 = no sub-LLMs) |
| `nested_max_iterations` | `10` | Max REPL iterations per nested sub-LLM |
| `nested_max_output_length` | `4096` | Max output chars from nested code execution |
| `**kwargs` | — | All standard `RLMEnv` parameters |

## Development

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT
