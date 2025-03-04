from datasets import Dataset


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def preprocess_dataset(dataset: Dataset) -> Dataset:
    return dataset.map(
        lambda x: {
            "prompt": [{"role": "user", "content": x["question"]}],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
