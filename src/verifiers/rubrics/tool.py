from typing import Dict, List

from verifiers.parsers import XMLParser
from verifiers.rubrics.rubric import Rubric


def make_tool_use_reward_func(
    assistant_parser: XMLParser = XMLParser(fields=["think", ("tool", "answer")]),
    env_parser: XMLParser = XMLParser(fields=["result"]),
):
    def tool_execution_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """

        def check_execution(trajectory):
            tool_attempts = 0
            successful_executions = 0

            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg["role"] == "assistant":
                    # Use parser to check for tool tag
                    parsed = assistant_parser.parse(msg["content"])
                    if hasattr(parsed, "tool") and parsed.tool is not None:
                        # Found a properly formatted tool message
                        if i + 1 < len(trajectory) and trajectory[i + 1]["role"] == "tool":
                            tool_attempts += 1
                            # Check response with env_parser
                            parsed_response = env_parser.parse(trajectory[i + 1]["content"])
                            if (
                                hasattr(parsed_response, "result")
                                and parsed_response.result is not None
                                and not parsed_response.result.startswith("Error:")
                            ):
                                successful_executions += 1

            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return 0.2 * (successful_executions / tool_attempts)

        return [check_execution(c) for c in completions]

    return tool_execution_reward_func


class ToolRubric(Rubric):
    def __init__(self):
        super().__init__(reward_funcs=[make_tool_use_reward_func()])
