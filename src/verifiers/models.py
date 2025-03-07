from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


class Input(TypedDict):
    prompt: list[Message]

class RunContext(TypedDict):
    input: Input
