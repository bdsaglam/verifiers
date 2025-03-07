import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm  # Import tqdm for progress bar

import verifiers as vf
from verifiers.imports import LLM, SamplingParams
from verifiers.parsers.xml_parser import XMLParser
from verifiers.prompts import CALCULATOR_FEW_SHOT, CODE_FEW_SHOT

load_dotenv()

log = logging.getLogger(__name__)

app = typer.Typer()


def prepare_dataset(dataset_path: str, dataset_name: str, split: str) -> Dataset:
    ds = load_dataset(dataset_path, dataset_name, split=split)

    if "gsm8k" in dataset_path:
        from verifiers.datasets.gsm8k import preprocess_dataset

        ds = preprocess_dataset(ds)
    else:
        raise ValueError(f"Dataset {dataset_path} not supported")

    return ds


def create_environment(
    env_type: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: Any,
):
    """
    Create and initialize the appropriate environment based on the specified type.

    Args:
        env_type: Type of environment to create ('code' or 'tool')
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        tokenizer: Tokenizer instance

    Returns:
        A tuple containing the initialized environment and a default suffix for run naming
    """
    if env_type.startswith("code"):
        log.info("Initializing CodeEnv environment")

        executor = env_type.split("/", 1)[-1] or "docker"
        if executor == "e2b":
            from verifiers.codex.e2b import E2BPythonExecutor

            code_executor = E2BPythonExecutor()
        elif executor == "docker":
            from verifiers.codex.docker import DockerPythonExecutor

            code_executor = DockerPythonExecutor()
        elif executor == "local":
            from verifiers.codex.local import LocalPythonExecutor

            code_executor = LocalPythonExecutor()
        else:
            raise ValueError(f"Unknown executor: {executor}")

        assistant_parser = XMLParser(fields=["think", ("code", "answer")])
        env_parser = XMLParser(fields=["output"])

        vf_env = vf.CodeEnv(
            code_executor=code_executor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            few_shot=CODE_FEW_SHOT[0],
            few_shot_prob=1.0,
            n_jobs=1,
        )
    elif env_type.lower() == "tool":
        log.info("Initializing ToolEnv environment")
        from verifiers.tools import calculator

        assistant_parser = XMLParser(fields=["think", ("tool", "answer")])
        env_parser = XMLParser(fields=["result"])

        vf_env = vf.ToolEnv(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            few_shot=CALCULATOR_FEW_SHOT[0],
            few_shot_prob=1.0,
            tools=[calculator],
            assistant_parser=assistant_parser,
            env_parser=env_parser,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}. Choose 'code' or 'tool'.")

    return vf_env


@app.command("predict")
def predict(
    model_name: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model"),
    env_type: str = typer.Option("code", "--env", help="Environment type: 'code' or 'tool'"),
    dataset_path: str = typer.Option("openai/gsm8k"),
    dataset_name: str = typer.Option("main"),
    dataset_split: str = typer.Option("test[:32]"),
    max_completion_length: int = typer.Option(1024, "-cl"),
    batch_size: int = typer.Option(32, "-bs"),
    temperature: float = typer.Option(0.3, "-t"),
    top_p: float = typer.Option(0.95, "-p"),
    out: Path = typer.Option("./outputs/", "--out"),
    report_to: str = typer.Option("wandb"),
    seed: int = 89,
):
    """Predict with a model on a dataset with tool use or code generation."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = prepare_dataset(dataset_path, dataset_name, dataset_split)
    log.info(f"Dataset: {len(dataset)}")

    # Load model and tokenizer
    tokenizer = vf.get_tokenizer(model_name)

    # Initialize environment based on env_type
    env = create_environment(
        env_type=env_type,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Initialize vLLM for serving
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        seed=seed,
    )

    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_completion_length,
        stop=[
            *env.special_stop_tokens,
            tokenizer.eos_token,
            tokenizer.pad_token,
        ],
        temperature=temperature,
        top_p=top_p,
    )

    # Process dataset in batches with progress bar
    ds = env.get_dataset()
    records = []
    for i in tqdm(range(0, len(ds), batch_size), desc="Processing batches", total=len(ds) // batch_size):
        inputs = ds.select(range(i, min(i + batch_size, len(ds))))

        # Generate completions and interact with environment
        result = env.generate(
            inputs=inputs,
            llm=llm,
            sampling_params=sampling_params,
        )

        # Store trajectories
        for input_data, messages in zip(inputs, result["messages"]):
            record = {
                **input_data,
                "trajectory": input_data["prompt"] + messages,
            }
            records.append(record)

    # Save trajectories to jsonl file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_id = f"{dataset_path}-{dataset_name}-{dataset_split.split('[', 1)[0]}".replace("/", "-")
    output_file = out / f"predictions-{dataset_id}-{env_type}-{model_name.split('/')[-1]}-{timestamp}.jsonl"
    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    log.info(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    app()
