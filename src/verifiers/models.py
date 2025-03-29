from typing import Any, Callable, TypedDict

RewardFunc = Callable[[list, list], list[float]]


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
