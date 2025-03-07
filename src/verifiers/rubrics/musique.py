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
