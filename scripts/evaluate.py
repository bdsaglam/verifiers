import json
from pathlib import Path

import pandas as pd
import typer

from verifiers.metrics.musique import exact_match, f1
from verifiers.rubrics.utils import get_last_answer


def evaluate(filepath: Path = typer.Argument(), output_dir: Path = typer.Option(...)):
    df = pd.read_json(filepath, lines=True)
    df["n_hops"] = df["supporting_titles"].apply(len)
    df["predicted_answer"] = df["trajectory"].apply(get_last_answer)
    df["exact_match"] = df.apply(lambda row: exact_match(row["predicted_answer"], row["answers"]), axis=1)
    df["f1"] = df.apply(lambda row: f1(row["predicted_answer"], row["answers"]), axis=1)
    df[["id", "answers", "predicted_answer", "n_hops", "exact_match", "f1"]].to_json(
        output_dir / "results.jsonl", orient="records", lines=True
    )
    scores = df[["exact_match", "f1"]].mean().to_dict()
    with open(output_dir / "scores.json", "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    typer.run(evaluate)
