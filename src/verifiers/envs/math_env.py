from typing import Any, Dict, List, Tuple

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.simple_env import SimpleEnv
from verifiers.parsers import XMLParser
from verifiers.prompts import MATH_FEW_SHOT, SIMPLE_PROMPT
from verifiers.rubrics.format import make_format_reward_func, make_xml_reward_func
from verifiers.rubrics.math import int_answer_reward_func, numerical_equivalence_reward_func
from verifiers.utils import preprocess_dataset


class MathEnv(SimpleEnv):
    def __init__(
        self,
        dataset: str = "gsm8k",
        system_prompt: str = SIMPLE_PROMPT,
        few_shot: List[Dict[str, str]] = MATH_FEW_SHOT[0],
        fields: List[str | Tuple[str, ...]] = ["think", "answer"],
        **kwargs,
    ):
        super().__init__(system_prompt=system_prompt, few_shot=few_shot, **kwargs)
        self.parser = XMLParser(fields=fields)
        self.dataset_name = dataset
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="train",
            system_prompt=system_prompt,
            few_shot=few_shot,
        )
        self.eval_dataset = None
        self.reward_funcs = [
            numerical_equivalence_reward_func,
            int_answer_reward_func,
            make_xml_reward_func(self.parser),
            make_format_reward_func(self.parser),
        ]

    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset

    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Dataset | None:
        if self.eval_dataset is None:
            self.eval_dataset = preprocess_dataset(
                dataset_name=self.dataset_name, split="test", system_prompt=self.system_prompt, few_shot=self.few_shot
            )
        if n > 0:
            return self.eval_dataset.shuffle().select(range(n))  # type: ignore
        return self.eval_dataset

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
