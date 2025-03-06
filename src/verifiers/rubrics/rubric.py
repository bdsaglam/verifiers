import logging
from abc import ABC
from typing import List

from trl.trainer.grpo_trainer import RewardFunc

log = logging.getLogger(__name__)


class Rubric(ABC):
    def __init__(self, reward_funcs: List[RewardFunc], reward_weights: List[float] | None = None):
        self.reward_funcs = reward_funcs
        self.reward_weights = reward_weights

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs

    def get_reward_weights(self) -> List[float] | None:
        return self.reward_weights


