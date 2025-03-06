from typing import List

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.rubrics.format import get_format_reward_func, get_xml_reward_func
from verifiers.rubrics.qa import exact_answer_reward_func
from verifiers.rubrics.utils import get_last_answer


def int_answer_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the final answer is an integer."""
    responses = [get_last_answer(c) for c in completions]
    return [1.0 if str(r).isdigit() else 0.0 for r in responses]


def equals_reward_func(completions, answer, **kwargs) -> List[float]:
    responses = [c[0]["content"] for c in completions]
    return [1.0 if r == a else 0.0 for r, a in zip(responses, answer)]


class MathRubric(Rubric):
    def __init__(self, parser: XMLParser = XMLParser(fields=["think", "answer"])):
        super().__init(
            reward_funcs=[
                exact_answer_reward_func,
                int_answer_reward_func,
                get_xml_reward_func(parser),
                get_format_reward_func(parser),
            ]
        )
