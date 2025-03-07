import logging
from pathlib import Path
from typing import Any

import typer
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from trl import GRPOConfig

import verifiers as vf
from verifiers.envs.tool_env import ToolEnv
from verifiers.prompts import QA_TOOL_PROMPT_TEMPLATE, RETRIEVE_FEW_SHOT
from verifiers.rubrics.qa import musique_em_reward_func, musique_f1_reward_func
from verifiers.tools import make_retrieve_tool

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
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: Any,
    retriever: str,
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
        few_shot=RETRIEVE_FEW_SHOT[0],
        few_shot_prob=0.7,
        tools=[make_retrieve_tool(name=retriever, top_k=2)],
        system_prompt=QA_TOOL_PROMPT_TEMPLATE,
        max_steps=20,
    )

    return vf_env


def save_artifacts(model, tokenizer, model_id: str, hub_dir: Path):
    log.info(f"Saving model and tokenizer locally: {model_id}")
    model.save_pretrained(hub_dir / model_id)
    tokenizer.save_pretrained(hub_dir / model_id)


@app.command("train")
def train(
    model_name: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model"),
    dataset_path: str = typer.Option("bdsaglam/musique"),
    dataset_name: str = typer.Option("answerable"),
    dataset_split: str = typer.Option("train"),
    eval_dataset_path: str = typer.Option("bdsaglam/musique"),
    eval_dataset_name: str = typer.Option("answerable"),
    eval_dataset_split: str = typer.Option("validation[:32]"),
    max_prompt_length: int = typer.Option(4096, "-pl"),
    max_completion_length: int = typer.Option(1024, "-cl"),
    num_generations: int = typer.Option(8, "-g", help="Number of generations per prompt"),
    batch_size: int = typer.Option(32, "-bs", help="Per device batch size"),
    gradient_accumulation_steps: int = typer.Option(2, "-gacc"),
    learning_rate: float = typer.Option(1e-6, "-lr"),
    beta: float = typer.Option(0.04, "--beta", help="KL penalty coefficient"),
    eval_steps: int = typer.Option(100, "--eval-steps"),
    out: Path = typer.Option("./outputs/", "--out"),
    hub_dir: Path = typer.Option("/home/baris/.cache/huggingface/tgi/local"),
    suffix: str = typer.Option("grpo", "--suffix", help="Custom suffix for the run name"),
    retriever: str = typer.Option("bm25", "--retriever", help="Retriever to use"),
):
    """Train a model using GRPO for code generation or tool use."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    hub_dir = Path(hub_dir)
    hub_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_dataset = prepare_dataset(dataset_path, dataset_name, dataset_split)
    log.info(f"Train dataset: {len(train_dataset)}")

    eval_dataset = prepare_dataset(eval_dataset_path, eval_dataset_name, eval_dataset_split)
    log.info(f"Eval dataset: {len(eval_dataset)}")

    # Load model and tokenizer
    model, tokenizer = vf.get_model_and_tokenizer(model_name)

    # Initialize environment based on env_type
    vf_env = create_environment(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        retriever=retriever,
    )

    # Use provided suffix or default based on env_type
    run_name = f"tool-{model_name.split('/')[-1]}-{dataset_path.split('/')[-1]}-{suffix}"

    training_args = GRPOConfig(
        output_dir=out / run_name,
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
        run_name=run_name,
        reward_weights=None,
        eval_strategy="steps",
        eval_on_start=True,
        eval_steps=eval_steps,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=1,
    )
    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

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
    trainer.train()

    # Save artifacts if main process
    if accelerator.is_main_process:
        save_artifacts(model, tokenizer, run_name, hub_dir)


if __name__ == "__main__":
    app()
