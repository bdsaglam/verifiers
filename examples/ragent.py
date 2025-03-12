import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import typer
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from tqdm import tqdm
from trl import GRPOConfig

import verifiers as vf
import wandb
from verifiers.envs.tool_env import ToolEnv
from verifiers.imports import LLM, SamplingParams
from verifiers.prompts import QA_TOOL_PROMPT_TEMPLATE, RETRIEVE_FEW_SHOT
from verifiers.rubrics.musique import (
    musique_em_reward_func,
    musique_f1_reward_func,
)
from verifiers.tools import make_retrieve_tool
from verifiers.utils.model_utils import get_tokenizer

load_dotenv()

log = logging.getLogger(__name__)

app = typer.Typer()

accelerator = Accelerator()


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
    """
    Create and initialize the appropriate environment based on the specified type.

    Args:
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        tokenizer: Tokenizer instance
        retriever: Retriever to use

    Returns:
        A tuple containing the initialized environment and a default suffix for run naming
    """
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


def save_artifacts(model, tokenizer, model_id: str, hub_dir: Path):
    log.info(f"Saving model and tokenizer locally: {model_id}")
    model.save_pretrained(hub_dir / model_id)
    tokenizer.save_pretrained(hub_dir / model_id)


def get_model_name(model_path: str) -> str:
    if Path(model_path).exists():
        with open(Path(model_path) / "config.json", "r") as f:
            config = json.load(f)
        return config["_name_or_path"].split("/")[-1]
    else:
        return model_path.split("/")[-1]


@app.command("train")
def train(
    model_path: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model"),
    dataset_path: str = typer.Option("bdsaglam/musique"),
    dataset_name: str = typer.Option("answerable"),
    dataset_split: str = typer.Option("train"),
    eval_dataset_path: str = typer.Option("bdsaglam/musique"),
    eval_dataset_name: str = typer.Option("answerable"),
    eval_dataset_split: str = typer.Option("validation[:32]"),
    retriever: str = typer.Option("lexical", "--retriever", help="Retriever to use"),
    retriever_top_k: int = typer.Option(2, "--retriever-top-k", help="Number of retriever results to use"),
    few_shot_prob: float = typer.Option(1.0, "--few-shot-prob", help="Probability of using few-shot examples"),
    n_env_jobs: int = typer.Option(32, "--n-env-jobs", help="Number of environments to run in parallel"),
    max_prompt_length: int = typer.Option(4096, "-pl"),
    max_completion_length: int = typer.Option(1024, "-cl"),
    num_generations: int = typer.Option(8, "-g", help="Number of generations per prompt"),
    batch_size: int = typer.Option(32, "-bs", help="Per device batch size"),
    gradient_accumulation_steps: int = typer.Option(4, "-gacc"),
    learning_rate: float = typer.Option(1e-6, "-lr"),
    peft: bool = typer.Option(True, help="Use PEFT"),
    lora_r: int = typer.Option(32, help="LORA rank"),
    lora_alpha: int = typer.Option(64, help="LORA alpha"),
    eval_strategy: str = typer.Option("no", help="Evaluation strategy"),
    eval_on_start: bool = typer.Option(False, help="Evaluate on start"),
    eval_steps: int = typer.Option(100, help="Evaluation steps"),
    report_to: str = typer.Option("wandb", help="Report to wandb"),
    push_to_hub: bool = typer.Option(True, help="Push to hub"),
    out: Path = typer.Option("./outputs/"),
    resume_from_checkpoint: bool = typer.Option(False, help="Resume training from a checkpoint"),
):
    """Train a model using GRPO for code generation or tool use."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_dataset = prepare_dataset(dataset_path, dataset_name, dataset_split)
    log.info(f"Train dataset: {len(train_dataset)}")

    eval_dataset = prepare_dataset(eval_dataset_path, eval_dataset_name, eval_dataset_split)
    log.info(f"Eval dataset: {len(eval_dataset)}")

    # Load model and tokenizer
    model, tokenizer = vf.get_model_and_tokenizer(model_path)

    # Initialize environment based on env_type
    vf_env = create_environment(
        tokenizer=tokenizer,
        retriever=retriever,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        n_jobs=n_env_jobs,
        top_k=retriever_top_k,
        few_shot_prob=few_shot_prob,
    )

    # Use provided suffix or default based on env_type
    run_name = f"ragent-{get_model_name(model_path)}-{dataset_path.split('/')[-1]}-grpo"

    training_args = GRPOConfig(
        output_dir=out / run_name,
        push_to_hub=push_to_hub,
        hub_model_id=run_name,
        bf16=True,
        num_train_epochs=1,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.01,
        num_iterations=2,  # steps per global batch (1 on-policy, 1 off-policy)
        num_generations=num_generations,
        beta=0.04,
        reward_weights=None,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.7,
        save_strategy="steps",
        save_steps=100,
        save_only_model=False,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to=report_to,
        run_name=run_name,
        eval_strategy=eval_strategy,
        eval_on_start=eval_on_start,
        eval_steps=eval_steps,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=1,
    )
    # Configure LoRA
    if peft:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Initialize trainer
    reward_funcs = [
        musique_em_reward_func,
        musique_f1_reward_func,
        *vf_env.get_reward_funcs(),
    ]
    trainer = vf.GRPOEnvTrainer(
        model=model,
        peft_config=peft_config,
        env=vf_env,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=vf_env.get_dataset(),
        eval_dataset=vf_env.get_eval_dataset(),
    )

    # Start training
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Cleanup
    wandb.finish()
    del model
    del trainer
    torch.cuda.empty_cache()


@app.command("predict")
def predict(
    model_path: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model"),
    dataset_path: str = typer.Option("bdsaglam/musique-mini"),
    dataset_name: str = typer.Option("answerable"),
    dataset_split: str = typer.Option("validation"),
    retriever: str = typer.Option("lexical", "--retriever", help="Retriever to use"),
    retriever_top_k: int = typer.Option(2, "--retriever-top-k", help="Number of retriever results to use"),
    few_shot_prob: float = typer.Option(1.0, "--few-shot-prob", help="Probability of using few-shot examples"),
    n_env_jobs: int = typer.Option(32, "--n-env-jobs", help="Number of environments to run in parallel"),
    max_completion_length: int = typer.Option(1024, "-cl"),
    batch_size: int = typer.Option(32, "-bs"),
    temperature: float = typer.Option(0.3, "-t"),
    top_p: float = typer.Option(0.95, "-p"),
    out: Path = typer.Option("./outputs/", "--out"),
    seed: int = 89,
):
    """Predict with a model on a dataset using RAG-based verification."""

    # Load dataset
    dataset = prepare_dataset(dataset_path, dataset_name, dataset_split)
    log.info(f"Dataset: {len(dataset)}")

    # Load model and tokenizer
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
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        seed=seed,
    )

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
        inputs = ds.select(range(i, min(i + batch_size, len(ds))))

        # Generate completions and interact with environment
        result = vf_env.generate(
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
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_id = f"{dataset_path}-{dataset_name}-{dataset_split.split('[', 1)[0]}".replace("/", "-")
    output_file = out / f"predictions-{dataset_id}-{model_path.split('/')[-1]}-{timestamp}.jsonl"
    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    log.info(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    app()
