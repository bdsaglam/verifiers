from typing import Dict, List

from verifiers.metrics import musique
from verifiers.rubrics.utils import get_last_answer


def exact_answer_reward_func(completions, answer, **kwargs) -> List[float]:
    """Reward function that checks if the final answer matches the expected answer."""
    responses = [get_last_answer(c) or "" for c in completions]
    return [1.0 if str(r).strip() == str(a).strip() else 0.0 for r, a in zip(responses, answer)]


def musique_em_reward_func(
    completions: List[List[Dict[str, str]]],
    answers: list[list[str]],
    **kwargs,
) -> List[float]:
    """
    Reward function that checks question answering success.
    """
    predicted_answers = [get_last_answer(c) or "" for c in completions]
    return [
        musique.exact_match(predicted_answer, references)
        for predicted_answer, references in zip(predicted_answers, answers)
    ]


def musique_f1_reward_func(
    completions: List[List[Dict[str, str]]],
    answers: list[list[str]],
    **kwargs,
) -> List[float]:
    predicted_answers = [get_last_answer(c) or "" for c in completions]
    return [
        musique.f1(predicted_answer, references) for predicted_answer, references in zip(predicted_answers, answers)
    ]
