from datasets import Dataset


def _make_doc(p: dict) -> dict:
    return {
        "id": p["idx"],
        "text": f"# {p['title']}\n{p['paragraph_text']}",
        "is_supporting": p["is_supporting"],
    }


def preprocess_dataset(dataset: Dataset) -> Dataset:
    dataset = dataset.map(
        lambda x: {
            "prompt": [{"role": "user", "content": x["question"]}],
            "docs": [_make_doc(p) for p in x["paragraphs"]],
            "answer": x["answer"],
            "answers": [x["answer"], *x["answer_aliases"]],
            "supporting_titles": [p["title"] for p in x["paragraphs"] if p["is_supporting"]],
        },
        remove_columns=[
            "question",
            "paragraphs",
            "question_decomposition",
            "answerable",
            "answer_aliases",
        ],
    )
    return dataset
