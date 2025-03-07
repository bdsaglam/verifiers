from typing import Callable, TypedDict, Union

from transformers import PreTrainedModel
from trl.trainer.grpo_trainer import RewardFunc


class Message(TypedDict):
    role: str
    content: str


class Input(TypedDict):
    prompt: list[Message]


class RunContext(TypedDict):
    input: Input


__all__ = [
    "Input",
    "Message",
    "RewardFunc",
    "RunContext",
]
