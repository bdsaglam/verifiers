import json
from pathlib import Path
from typing import Any, Dict, Set

import pandas as pd
import typer

from verifiers.metrics.musique import exact_match, f1
from verifiers.rubrics.utils import get_last_answer
from verifiers.tools.retrieve import extract_all_retrieved_doc_ids


def calculate_supporting_metrics(retrieved: Set[str], supporting: Set[str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 for supporting titles.

    Args:
        retrieved: Set of retrieved titles
        supporting: Set of supporting titles

    Returns:
        Dictionary containing precision, recall, and F1 scores
    """
    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not supporting:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = len(retrieved & supporting)
    precision = intersection / len(retrieved) if retrieved else 0.0
    recall = intersection / len(supporting) if supporting else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1_score}


def process_row(row: pd.Series) -> Dict[str, Any]:
    """Process a single row of data to compute all metrics.

    Args:
        row: DataFrame row containing trajectory, supporting_titles, and answers

    Returns:
        Dictionary containing all computed metrics
    """
    # Extract data
    predicted_answer = get_last_answer(row["trajectory"])
    supporting_doc_ids = set(doc["idx"] for doc in row["docs"] if doc["is_supporting"])
    retrieved_doc_ids = set(extract_all_retrieved_doc_ids(row["trajectory"]))

    # Calculate answer metrics
    answer_exact_match = exact_match(predicted_answer, row["answers"])
    answer_f1 = f1(predicted_answer, row["answers"])

    # Calculate supporting metrics
    supporting_metrics = calculate_supporting_metrics(retrieved_doc_ids, supporting_doc_ids)

    return {
        "predicted_answer": predicted_answer,
        "exact_match": answer_exact_match,
        "f1": answer_f1,
        "supporting_doc_ids": list(supporting_doc_ids),
        "retrieved_doc_ids": list(retrieved_doc_ids),
        "supporting.precision": supporting_metrics["precision"],
        "supporting.recall": supporting_metrics["recall"],
        "supporting.f1": supporting_metrics["f1"],
    }


def evaluate(filepath: Path = typer.Argument(), output_dir: Path = typer.Option(...)):
    """Evaluate model performance and save results.

    Args:
        filepath: Path to the input JSON Lines file
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_json(filepath, lines=True)

    # Process all metrics in a single pass
    metrics_df = pd.DataFrame(df.apply(process_row, axis=1).tolist())
    result_df = pd.concat([df[["id", "n_hops", "answers", "supporting_doc_slugs"]], metrics_df], axis=1)

    # Select columns for output
    columns = [
        "id",
        "n_hops",
        "docs",
        # QA
        "answers",
        "predicted_answer",
        "exact_match",
        "f1",
        # Retrieval
        "supporting_doc_ids",
        "retrieved_doc_ids",
        "supporting.precision",
        "supporting.recall",
        "supporting.f1",
    ]

    # Save results
    result_df[columns].to_json(output_dir / "results.jsonl", orient="records", lines=True)

    # Calculate and save aggregate scores
    score_columns = ["exact_match", "f1", "supporting.precision", "supporting.recall", "supporting.f1"]
    scores = result_df[score_columns].mean().to_dict()

    with open(output_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    typer.run(evaluate)
