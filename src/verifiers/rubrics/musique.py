import re
from typing import List

from verifiers.metrics.musique import exact_match, f1
from verifiers.models import Message
from verifiers.rubrics.utils import get_last_answer


def musique_em_reward_func(
    completions: List[List[Message]],
    answers: list[list[str]],
    **kwargs,
) -> List[float]:
    """
    Reward function that checks question answering success.
    """
    predicted_answers = [get_last_answer(c) or "" for c in completions]
    return [
        exact_match(predicted_answer, references) for predicted_answer, references in zip(predicted_answers, answers)
    ]


def musique_f1_reward_func(
    completions: List[List[Message]],
    answers: list[list[str]],
    **kwargs,
) -> List[float]:
    predicted_answers = [get_last_answer(c) or "" for c in completions]
    return [f1(predicted_answer, references) for predicted_answer, references in zip(predicted_answers, answers)]


def _extract_retrieved_titles(content: str) -> list[str]:
    return [title.strip() for title in re.findall(r"^# (.*)", content, re.MULTILINE)]


def musique_supporting_f1_reward_func(
    completions: List[List[Message]],
    supporting_titles: list[list[str]],
    **kwargs,
) -> List[float]:
    rewards = []
    for completion in completions:
        tool_messages = [msg for msg in completion if msg["role"] == "tool"]
        retrieved_titles = [_extract_retrieved_titles(msg["content"]) for msg in tool_messages]
        precision = len(set(retrieved_titles) & set(supporting_titles)) / len(retrieved_titles)
        recall = len(set(retrieved_titles) & set(supporting_titles)) / len(supporting_titles)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        rewards.append(f1)
    return rewards
