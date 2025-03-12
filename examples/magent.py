import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from tqdm import tqdm
from trl import GRPOConfig

import verifiers as vf
from verifiers.imports import LLM, SamplingParams
from verifiers.parsers.xml_parser import XMLParser
from verifiers.prompts import CALCULATOR_FEW_SHOT, CODE_FEW_SHOT
from verifiers.rubrics.math import int_answer_reward_func, numerical_equivalence_reward_func

load_dotenv()

log = logging.getLogger(__name__)

app = typer.Typer()

accelerator = Accelerator()


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
            n_jobs=32,
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
            n_jobs=32,
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}. Choose 'code' or 'tool'.")

    return vf_env


def save_artifacts(model, tokenizer, model_id: str, hub_dir: Path):
    log.info(f"Saving model and tokenizer locally: {model_id}")
    model.save_pretrained(hub_dir / model_id)
    tokenizer.save_pretrained(hub_dir / model_id)


@app.command("train")
def train(
    model_name: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model"),
    env_type: str = typer.Option("code", "--env", help="Environment type: 'code' or 'tool'"),
    dataset_path: str = typer.Option("openai/gsm8k"),
    dataset_name: str = typer.Option("main"),
    dataset_split: str = typer.Option("train"),
    eval_dataset_path: str = typer.Option("openai/gsm8k"),
    eval_dataset_name: str = typer.Option("main"),
    eval_dataset_split: str = typer.Option("test[:32]"),
    max_prompt_length: int = typer.Option(1024, "-pl"),
    max_completion_length: int = typer.Option(1024, "-cl"),
    num_generations: int = typer.Option(8, "-g", help="Number of generations per prompt"),
    batch_size: int = typer.Option(32, "-bs", help="Per device batch size"),
    gradient_accumulation_steps: int = typer.Option(1, "-gacc"),
    learning_rate: float = typer.Option(1e-6, "-lr"),
    beta: float = typer.Option(0.04, "--beta", help="KL penalty coefficient"),
    eval_steps: int = typer.Option(100, "--eval-steps"),
    out: Path = typer.Option("./outputs/", "--out"),
    hub_dir: Path = typer.Option("/home/baris/.cache/huggingface/tgi/local"),
    suffix: str = typer.Option("grpo", "--suffix", help="Custom suffix for the run name"),
    report_to="wandb",
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
        env_type=env_type,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Use provided suffix or default based on env_type
    run_name = f"{env_type}-{model_name.split('/')[-1]}-{dataset_path.split('/')[-1]}-{suffix}"

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
        report_to=report_to,
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
        int_answer_reward_func,
        numerical_equivalence_reward_func,
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
