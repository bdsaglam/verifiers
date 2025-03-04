import gc
import logging
from pathlib import Path
from typing import Optional

import torch
import typer
from accelerate import Accelerator

import verifiers as vf
from verifiers.prompts import CALCULATOR_FEW_SHOT
from verifiers.tools import calculator
from trl import GRPOConfig
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

app = typer.Typer()

accelerator = Accelerator()


def clear_memory():
    """Clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()


def prepare_dataset(env):
    """Prepare the GSM8K dataset for training"""
    dataset = env.get_dataset()
    eval_dataset = env.get_eval_dataset(n=100)
    return dataset, eval_dataset


def save_artifacts(model, tokenizer, model_id: str, hub_dir: Path):
    # Save locally
    log.info(f"Saving model and tokenizer locally: {model_id}")
    model.save_pretrained(hub_dir / model_id)
    tokenizer.save_pretrained(hub_dir / model_id)


@app.command("train")
def train(
    model_name: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model"),
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

    # Set up logging
    logging.basicConfig(level=log_level)

    # Load model and tokenizer
    model, tokenizer = vf.get_model_and_tokenizer(model_name)

    # Initialize tool environment
    vf_env = vf.ToolEnv(
        tokenizer=tokenizer,
        dataset="gsm8k",
        few_shot=CALCULATOR_FEW_SHOT[0],
        tools=[calculator],
    )

    # Prepare datasets
    dataset, eval_dataset = prepare_dataset(vf_env)
    rubric = vf_env.get_rubric()

    # Configure training arguments
    run_name = f"{model_name.split('/')[-1]}-{suffix}"
    output_dir = out / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        num_train_epochs=1,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.01,
        num_iterations=2,
        beta=beta,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=batch_size,
        num_generations=num_generations,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=100,
        save_only_model=True,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.7,
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
        reward_funcs=rubric,
        env=vf_env,
        args=training_args,
        train_dataset=dataset,
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
