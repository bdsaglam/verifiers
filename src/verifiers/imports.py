"""Central import handling for platform-specific dependencies."""

import platform

# Use mock vLLM on macOS, real vLLM otherwise
if platform.system() == "Darwin":
    from .mock_vllm import LLM, CompletionOutput, SamplingParams
else:
    from vllm import LLM, CompletionOutput, SamplingParams  # type: ignore

__all__ = ["LLM", "SamplingParams", "CompletionOutput"]
