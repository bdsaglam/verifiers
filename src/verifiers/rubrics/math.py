from typing import List

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.rubrics.format import make_format_reward_func, make_xml_reward_func
from verifiers.rubrics.qa import exact_answer_reward_func
from verifiers.rubrics.utils import get_last_answer


def int_answer_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the final answer is an integer."""
    responses = [get_last_answer(c) for c in completions]
    return [1.0 if str(r).strip().isdigit() else 0.0 for r in responses]


def numeric_answer_reward_func(completions, **kwargs) -> List[float]:
    responses = [get_last_answer(c) for c in completions]
    return [1.0 if str(r).strip().isnumeric() else 0.0 for r in responses]


def safe_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return None


def numerical_equivalence_reward_func(completions, answer, **kwargs) -> List[float]:
    responses = [get_last_answer(c) for c in completions]
    return [1.0 if safe_float(r) == a else 0.0 for r, a in zip(responses, answer)]


class MathRubric(Rubric):
    def __init__(self, parser: XMLParser = XMLParser(fields=["think", "answer"])):
        super().__init(
            reward_funcs=[
                exact_answer_reward_func,
                int_answer_reward_func,
                make_xml_reward_func(parser),
                make_format_reward_func(parser),
            ]
        )
