import random

from datasets import Dataset


def update_messages(
    messages: list[dict],
    system_prompt: str = None,
    few_shot: list[dict] = None,
    fewshot_prob: float = 0.0,
) -> list[dict]:
    # Order: system prompt, few-shot examples (user, assistant, tool, etc.), user message
    suffix = []
    if messages[-1]["role"] == "user":
        suffix.append(messages.pop())

    # Insert system prompt if it doesn't exist
    if system_prompt:
        if "system" in [m["role"] for m in messages]:
            assert messages[0]["role"] != "system", "System prompt already exists"
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Add few-shot examples if they exist
    if few_shot and random.random() < fewshot_prob:
        messages.extend(few_shot)

    messages.extend(suffix)

    return messages


def prepare_example_for_env(
    example: dict,
    system_prompt: str = None,
    few_shot: list[dict] = None,
    fewshot_prob: float = 0.0,
) -> dict:
    return {
        "prompt": update_messages(example["prompt"], system_prompt, few_shot, fewshot_prob),
    }


def prepare_dataset_for_env(
    dataset: Dataset,
    system_prompt: str = None,
    few_shot: list[dict] = None,
    fewshot_prob: float = 0.0,
) -> Dataset:
    return dataset.map(lambda x: prepare_example_for_env(x, system_prompt, few_shot, fewshot_prob))
