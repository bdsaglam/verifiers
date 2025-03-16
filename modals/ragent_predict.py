import json
import logging
import os
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Union

import modal
from datasets import Dataset, load_dataset
from tqdm import tqdm

from verifiers.envs.tool_env import ToolEnv
from verifiers.imports import LLM, SamplingParams
from verifiers.prompts import QA_TOOL_PROMPT_TEMPLATE, RETRIEVE_FEW_SHOT
from verifiers.tools import make_retrieve_tool
from verifiers.utils.cuda import get_half_precision_dtype
from verifiers.utils.model_utils import get_tokenizer

log = logging.getLogger(__name__)

# Constants
MINUTES = 60
HOURS = 60 * MINUTES

# Define the volumes
pretrained_volume = modal.Volume.from_name("pretrained-vol", create_if_missing=True)
runs_volume = modal.Volume.from_name("runs-vol", create_if_missing=True)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
}

# GPU Configuration
GPU_CONFIG = os.environ.get("GPU_CONFIG", "a100:1")
if len(GPU_CONFIG.split(":")) <= 1:
    N_GPUS = int(os.environ.get("N_GPUS", 1))
    GPU_CONFIG = f"{GPU_CONFIG}:{N_GPUS}"

# Define the Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject(
        pyproject_toml="pyproject.toml",
        extra_options="retrieve",
        gpu=GPU_CONFIG,
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "WANDB_PROJECT": "agent-rl",
        }
    )
    .entrypoint([])
)

# Define the app
app = modal.App("ragent-predict", image=image)


def prepare_dataset(dataset_path: str, dataset_name: str, split: str) -> Dataset:
    ds = load_dataset(dataset_path, dataset_name, split=split)

    if "musique" in dataset_path:
        from verifiers.datasets.musique import preprocess_dataset

        ds = preprocess_dataset(ds)
    else:
        raise ValueError(f"Dataset {dataset_path} not supported")

    return ds


def create_environment(
    tokenizer: Any,
    retriever: str,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    n_jobs: int = 1,
    top_k: int = 2,
    few_shot_prob: float = 1.0,
):
    """Create and initialize the appropriate environment based on the specified type."""
    log.info("Initializing the environment")

    vf_env = ToolEnv(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        tools=[make_retrieve_tool(name=retriever, top_k=top_k)],
        system_prompt=QA_TOOL_PROMPT_TEMPLATE,
        few_shot=RETRIEVE_FEW_SHOT[0],
        few_shot_prob=few_shot_prob,
        max_steps=20,
        n_jobs=n_jobs,
    )

    return vf_env


def get_model_name(model_path: str) -> str:
    if Path(model_path).exists():
        with open(Path(model_path) / "config.json", "r") as f:
            config = json.load(f)
        return config["_name_or_path"].split("/")[-1]
    else:
        return model_path.split("/")[-1]


@app.function(
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=4 * HOURS,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def predict(
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    dataset_path: str = "bdsaglam/musique-mini",
    dataset_name: str = "answerable",
    dataset_split: str = "validation",
    retriever: str = "bm25",
    retriever_top_k: int = 2,
    few_shot_prob: float = 1.0,
    n_env_jobs: int = 1,
    batch_size: int = 32,
    max_completion_length: int = 1024,
    temperature: float = 0.3,
    top_p: float = 0.95,
    seed: int = 89,
):
    """Predict with a model on a dataset using RAG-based verification."""
    if n_env_jobs > 1 and retriever == "bm25":
        raise ValueError("BM25 does not support parallel environments. Run rerank service and use 'lexical', instead.")

    # Ensure volumes contain latest files
    VOLUME_CONFIG["/pretrained"].reload()
    VOLUME_CONFIG["/runs"].reload()

    # Create run directory
    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"predict-{time_string}"
    run_folder = Path("/runs") / run_name
    run_folder.mkdir(parents=True, exist_ok=True)
    print(f"Starting prediction run in {run_folder}")

    try:
        # Load dataset
        dataset = prepare_dataset(dataset_path, dataset_name, dataset_split)
        print(f"Dataset loaded: {len(dataset)} examples")

        # Load model and tokenizer
        print(f"Loading model and tokenizer: {model_path}")
        tokenizer = get_tokenizer(model_path)

        # Initialize environment
        vf_env = create_environment(
            tokenizer=tokenizer,
            retriever=retriever,
            train_dataset=dataset,
            n_jobs=n_env_jobs,
            top_k=retriever_top_k,
            few_shot_prob=few_shot_prob,
        )

        # Initialize vLLM for serving
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype=get_half_precision_dtype(),
            gpu_memory_utilization=0.8,
            seed=seed,
        )
        print("Model initialized successfully")
        VOLUME_CONFIG["/pretrained"].commit()

        # Set up sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_completion_length,
            stop=[
                *vf_env.special_stop_tokens,
                tokenizer.eos_token,
                tokenizer.pad_token,
            ],
            temperature=temperature,
            top_p=top_p,
        )

        # Process dataset in batches with progress bar
        ds = vf_env.get_dataset()
        records = []
        for i in tqdm(range(0, len(ds), batch_size), desc="Processing batches", total=len(ds) // batch_size):
            batch_start = i
            batch_end = min(i + batch_size, len(ds))
            print(f"Processing batch {batch_start}-{batch_end}/{len(ds)}")

            try:
                inputs = ds.select(range(batch_start, batch_end))
                result = vf_env.generate(
                    inputs=inputs,
                    llm=llm,
                    sampling_params=sampling_params,
                )

                for input_data, messages in zip(inputs, result["messages"]):
                    record = {
                        **input_data,
                        "trajectory": input_data["prompt"] + messages,
                    }
                    records.append(record)
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {str(e)}")
                continue

        # Upload predictions to HuggingFace
        pred_ds = Dataset.from_list(records)
        pred_ds_id = f"{dataset_path}-predictions-{get_model_name(model_path)}-{time_string}".replace("/", "-")
        pred_ds.push_to_hub(pred_ds_id)
        print(f"Predictions uploaded to HuggingFace dataset: {pred_ds_id}")

        # Commit changes to volume
        VOLUME_CONFIG["/runs"].commit()

        return {
            "run_folder": str(run_folder),
            "pred_ds_id": pred_ds_id,
            "n_processed": len(records),
            "n_total": len(dataset),
        }

    except Exception as e:
        print(f"Run failed with error: {str(e)}")
        raise


@app.local_entrypoint()
def main(
    model_path: str = "Qwen/Qwen2.5-7B-Instruct",
    retriever: str = "bm25",
    retriever_top_k: int = 2,
    batch_size: int = 32,
):
    """Local entrypoint for running predictions."""
    result = predict.remote(
        model_path=model_path,
        retriever=retriever,
        retriever_top_k=retriever_top_k,
        batch_size=batch_size,
    )
    print(f"Dataset ID: {result['pred_ds_id']}")
    print(f"Processed {result['n_processed']}/{result['n_total']} examples")
