## Setup

```sh
conda activate verifiers
pip install -r requirements.txt
```

```sh
export LD_LIBRARY_PATH=/home/baris/miniconda3/envs/verifiers/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

## Train

```sh
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --config-file configs/zero3.yaml --num-processes 1 examples/gsm8k_calculator.py
```

```sh
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --config-file configs/zero3.yaml --num-processes 1 scripts/gsm8k_calculator.py train
```

```sh
huggingface-cli upload --repo-type model \
    Qwen2.5-1.5B-Instruct-gsm8k-calc \
    ./outputs/Qwen2.5-1.5B-Instruct-gsm8k-calc
```

## Predict

```sh
python scripts/merge.py \
    outputs/tool-Qwen2.5-1.5B-Instruct-gsm8k-grpo/checkpoint-900 \
    --out outputs/bdsaglam/Qwen2.5-1.5B-Instruct-GRPO-gsm8k-calc
```

```sh
python examples/predict.py \
    --model outputs/bdsaglam/Qwen2.5-1.5B-Instruct-GRPO-gsm8k-calc \
    --env tool \
    --dataset-path openai/gsm8k \
    --dataset-name main \
    --dataset-split 'test[:32]' \
    --report-to none
```
