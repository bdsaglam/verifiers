
accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 1 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --few-shot-prob 0.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 1 \
    --batch-size 6 \
    --num-generations 6 \
    --gradient-accumulation-steps 16 \
    2>&1 | tee tmp/logs/train-$(date +%s).log


accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 1 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --run-name 'Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250407_134926' \
    --few-shot-prob 0.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 1 \
    --batch-size 6 \
    --num-generations 6 \
    --gradient-accumulation-steps 8 \
    --resume-from-checkpoint \
    2>&1 | tee tmp/logs/train-$(date +%s).log


```sh

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --datasets 'bdsaglam/musique,answerable,train' \
    --few-shot-prob 0.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 16 \
    2>&1 | tee tmp/ragent-llama3-8b-round-2-$(date +%s).log


accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --run-name 'Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250329_222430' \
    --few-shot-prob 0.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 16 \
    --resume-from-checkpoint \
    2>&1 | tee tmp/ragent-llama3-8b-round-2-$(date +%s).log

```

```sh
export CUDA_VISIBLE_DEVICES=0,1

python scripts/ragent2.py train \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    2>&1 | tee tmp/logs/train-$(date +%s).log

```

# 2025-04-11

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

# 2025-04-17

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --noise-rate 1.0 \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.5 \
    --n-env-jobs 32 \
    --batch-size 24 \
    --num-generations 12 \
    --gradient-accumulation-steps 8 \
    --lora-r 32 \
    --lora-alpha 64 \
    2>&1 | tee tmp/logs/train-$(date +%s).log


# 2025-04-20

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 24 \
    --batch-size 24 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --n-epochs 4 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

python scripts/merge.py \
    outputs/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250421_000014/checkpoint-400 \
    --out outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-400

python scripts/merge.py \
    outputs/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250421_000014/checkpoint-1600 \
    --out outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1600

python scripts/merge.py \
    outputs/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250421_000014/checkpoint-1900 \
    --out outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1900

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'Qwen/Qwen2.5-14B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 16 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    --n-epochs 2 \
    2>&1 | tee tmp/logs/train-$(date +%s).log



## 2025-04-27

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --n-epochs 4 \
    --lora-r 64 \
    --lora-alpha 64 \
    2>&1 | tee tmp/logs/train-$(date +%s).log