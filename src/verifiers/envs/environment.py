import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.models import Input, Message

from ..imports import LLM, SamplingParams  # type: ignore


class Environment(ABC):
    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.tokenizer = None

    @abstractmethod
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        pass

    @abstractmethod
    def generate(
        self,
        inputs: List[Input],
        llm: LLM,
        sampling_params: SamplingParams,
        **kwargs: Any,
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Message]]]:
        pass
