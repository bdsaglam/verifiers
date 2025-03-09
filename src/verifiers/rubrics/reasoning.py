from typing import Callable, Dict, List

from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.utils import get_last_assistant_message


def make_reasoning_reward_func(
    think_tag: str = "think",
    min_length: int = 100,
    max_length: int = 1000,
) -> Callable:
    parser = XMLParser(fields=[think_tag])

    def reasoning_reward_func(
        completions: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        last_messages = [
            get_last_assistant_message(c) or "" for c in completions
        ]
        rewards = []
        for last_message in last_messages:
            parsed_response = parser.parse(last_message)
            if result := getattr(parsed_response, think_tag, None):
                # Reward based on the length of the think tag
                # Penalize too short or too long think tags
                if len(result) < min_length:
                    rewards.append(0.1)
                elif len(result) > max_length:
                    rewards.append(0.1)
                else:
                    rewards.append(0.2)
            else:
                rewards.append(0.0)
        return rewards

    return reasoning_reward_func
