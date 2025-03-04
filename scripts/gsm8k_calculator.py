import gc
import logging
from pathlib import Path
from typing import Optional

import torch
import typer
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from trl import GRPOConfig

import verifiers as vf
from verifiers.prompts import CALCULATOR_FEW_SHOT
from verifiers.tools import calculator

load_dotenv()

log = logging.getLogger(__name__)

app = typer.Typer()

accelerator = Accelerator()


def clear_memory():
    """Clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()


def prepare_dataset(dataset_path: str, dataset_name: str, split: str) -> Dataset:
    ds = load_dataset(dataset_path, dataset_name, split=split)

    if "gsm8k" in dataset_path:
        from verifiers.datasets.gsm8k import preprocess_dataset

        ds = preprocess_dataset(ds)
    else:
        raise ValueError(f"Dataset {dataset_path} not supported")

    return ds


def save_artifacts(model, tokenizer, model_id: str, hub_dir: Path):
    # Save locally
    log.info(f"Saving model and tokenizer locally: {model_id}")
    model.save_pretrained(hub_dir / model_id)
    tokenizer.save_pretrained(hub_dir / model_id)


@app.command("train")
def train(
    model_name: str = typer.Option("meta-llama/Llama-3.1-8B-Instruct", "--model"),
    dataset_path: str = typer.Option("openai/gsm8k"),
    dataset_name: str = typer.Option("main"),
    dataset_split: str = typer.Option("train[:128]"),
    eval_dataset_path: str = typer.Option("openai/gsm8k"),
    eval_dataset_name: str = typer.Option("main"),
    eval_dataset_split: str = typer.Option("test"),
    max_prompt_length: int = typer.Option(1024, "-pl"),
    max_completion_length: int = typer.Option(1024, "-cl"),
    num_generations: int = typer.Option(4, "-g", help="Number of generations per prompt"),
    batch_size: int = typer.Option(8, "-bs", help="Per device batch size"),
    gradient_accumulation_steps: int = typer.Option(4, "-gacc"),
    learning_rate: float = typer.Option(1e-6, "-lr"),
    beta: float = typer.Option(0.04, "--beta", help="KL penalty coefficient"),
    eval_steps: int = typer.Option(100, "--eval-steps"),
    eval_batch_size: int = typer.Option(8, "--eval-bs"),
    out: Path = typer.Option("./outputs/", "--out"),
    hub_dir: Path = typer.Option("/home/baris/.cache/huggingface/tgi/local"),
    log_level: str = "INFO",
    suffix: str = "gsm8k-calc",
):
    """Train a model using GRPO on the GSM8K dataset with calculator tool."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    hub_dir = Path(hub_dir)
    hub_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(level=log_level)

    # Load dataset
    train_dataset = prepare_dataset(dataset_path, dataset_name, dataset_split)
    eval_dataset = prepare_dataset(eval_dataset_path, eval_dataset_name, eval_dataset_split)

    # Load model and tokenizer
    model, tokenizer = vf.get_model_and_tokenizer(model_name)

    # Initialize tool environment
    vf_env = vf.ToolEnv(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        few_shot=CALCULATOR_FEW_SHOT[0],
        tools=[calculator],
    )

    # Configure training arguments
    run_name = f"{model_name.split('/')[-1]}-{suffix}"

    training_args = GRPOConfig(
        output_dir=out / run_name,
        run_name=run_name,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        num_train_epochs=1,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.01,
        num_iterations=2,  # steps per global batch (1 on-policy, 1 off-policy)
        beta=beta,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=batch_size,
        num_generations=num_generations,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.7,
        save_strategy="steps",
        save_steps=100,
        save_only_model=True,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
        reward_weights=None,
        eval_strategy="steps",
        eval_on_start=True,
        eval_steps=eval_steps,
        per_device_eval_batch_size=eval_batch_size,
        eval_accumulation_steps=1,
    )

    # Initialize trainer
    trainer = vf.GRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=vf_env.get_rubric(),
        env=vf_env,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()

    # Save artifacts if main process
    if accelerator.is_main_process:
        final_model_id = run_name
        save_artifacts(model, tokenizer, final_model_id, hub_dir)


@app.command("clear")
def clear():
    """Clear GPU memory"""
    log.info("Clearing GPU memory")
    clear_memory()


if __name__ == "__main__":
    app()
