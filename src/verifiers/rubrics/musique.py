import re
from typing import Generator, List

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


def extract_retrieved_titles(content: str) -> list[str]:
    return [title.strip() for title in re.findall(r"^# (.*)", content, re.MULTILINE)]


def extract_all_retrieved_titles(trajectory: list[Message]) -> Generator[str, None, None]:
    for msg in trajectory:
        if msg["role"] != "tool":
            continue
        yield from extract_retrieved_titles(msg["content"])


def musique_supporting_recall_reward_func(
    completions: List[List[Message]],
    supporting_titles: list[list[str]],
    **kwargs,
) -> List[float]:
    rewards = []
    for completion, _supporting_titles in zip(completions, supporting_titles):
        retrieved_titles = set(extract_all_retrieved_titles(completion))
        if len(retrieved_titles) == 0:
            rewards.append(0.0)
            continue
        recall = len(retrieved_titles & set(_supporting_titles)) / len(_supporting_titles)
        rewards.append(recall)
    return rewards


def musique_supporting_f1_reward_func(
    completions: List[List[Message]],
    supporting_titles: list[list[str]],
    **kwargs,
) -> List[float]:
    rewards = []
    for completion, _supporting_titles in zip(completions, supporting_titles):
        retrieved_titles = set(extract_all_retrieved_titles(completion))
        if len(retrieved_titles) == 0:
            rewards.append(0.0)
            continue
        precision = len(retrieved_titles & set(_supporting_titles)) / len(retrieved_titles)
        recall = len(retrieved_titles & set(_supporting_titles)) / len(_supporting_titles)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        rewards.append(f1)
    return rewards
