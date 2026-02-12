"""
verifiers-rlm: Variable recursion depth for the Recursive Language Model.

A lightweight extension for the verifiers library that adds configurable
recursion depth to RLMEnv, implementing the first item from Prime Intellect's
RLM roadmap.
"""

from .env import RecursiveRLMEnv
from .session import InProcessReplSession

__all__ = ["RecursiveRLMEnv", "InProcessReplSession"]
__version__ = "0.1.0"
