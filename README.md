# verifiers-rlm

Extensions for the verifiers RLM environment with configurable recursion depth and learned context compression.

## Packages

- **verifiers_rlm** - `RecursiveRLMEnv` with configurable `recursion_depth` (0=REPL only, 1=standard, N>1=nested sub-LLMs)
- **verifiers_context** - `ContextManagedRLMEnv` for learned context compression across conversation turns

## Install

```bash
pip install git+https://github.com/ascl1u/verifiers-rlm.git
```
