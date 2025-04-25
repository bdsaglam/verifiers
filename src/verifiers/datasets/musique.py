from datasets import Dataset


def _make_doc(p: dict) -> dict:
    return {
        "id": p["idx"],
        "text": f"# {p['title']}\n{p['paragraph_text']}",
        "title": p["title"],
        "is_supporting": p["is_supporting"],
    }


def preprocess_answer(answer: str) -> str:
    answer = answer.lower().strip()

    # Convert digits to numbers
    digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    if answer in digits:
        return str(digits.index(answer))

    # Convert ordinal numbers to numbers
    mapping = {
        "zeroth": "0th",
        "first": "1st",
        "second": "2nd",
        "third": "3rd",
        "fourth": "4th",
        "fifth": "5th",
        "sixth": "6th",
        "seventh": "7th",
        "eighth": "8th",
        "ninth": "9th",
    }
    for k, v in mapping.items():
        answer = answer.replace(k, v)
    return answer


def preprocess_example(x: dict) -> dict:
    answers = [x["answer"], *x["answer_aliases"]]
    answers += [preprocess_answer(a) for a in answers]
    supporting_titles = [p["title"] for p in x["paragraphs"] if p["is_supporting"]]
    return {
        "prompt": [{"role": "user", "content": x["question"]}],
        "docs": [_make_doc(p) for p in x["paragraphs"]],
        "answer": x["answer"],
        "answers": list(set(answers)),
        "supporting_titles": supporting_titles,
        "n_hops": len(supporting_titles),
    }


def preprocess_dataset(dataset: Dataset) -> Dataset:
    columns_to_remove = list(
        set(dataset.column_names) - {"id", "prompt", "docs", "answer", "answers", "supporting_titles", "n_hops"}
    )
    dataset = dataset.map(preprocess_example, remove_columns=columns_to_remove)
    return dataset
