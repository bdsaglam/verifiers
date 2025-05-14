## Setup

```sh
conda activate verifiers
pip install -r requirements.txt
```

```sh
export LD_LIBRARY_PATH=/home/baris/miniconda3/envs/verifiers/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

```sh
docker-compose down --remove-orphans; docker-compose up --build
```

## Train

```sh
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --repo-type model
```

```sh
export CUDA_VISIBLE_DEVICES=0,1,2
accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 2 \
    scripts/ragent.py train \
    2>&1 | tee tmp/logs/train-$(date +%s).log
```

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --model 'bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --datasets 'bdsaglam/musique,answerable,train;bdsaglam/hotpotqa-distractor,default,train' \
    --few-shot-prob 1.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 1 \
    2>&1 | tee tmp/logs/train-$(date +%s).log
```

### Publish manually
```sh
huggingface-cli upload --repo-type model \
    Qwen2.5-1.5B-Instruct-ragent-grpo-musique \
    ./outputs/Qwen2.5-1.5B-Instruct-ragent-grpo-musique
```


### Resume training

```sh
accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --model 'outputs/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --run-name 'Llama-3.1-8B-Instruct-ragent-grpo-musique-merged-ragent-grpo-20250329_222430' \
    --few-shot-prob 1.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 16 \
    --resume-from-checkpoint \
    2>&1 | tee tmp/logs/train-$(date +%s).log

```

## Predict

```sh
python scripts/merge.py \
    ./outputs/Qwen2.5-1.5B-Instruct-ragent-grpo-musique/checkpoint-1000 \
    --out outputs/Qwen2.5-1.5B-Instruct-ragent-grpo-musique-merged
```

```sh
# export MODEL=openai/meta-llama/Llama-3.3-70B-Instruct-Turbo
# export MODEL=Qwen/Qwen2.5-32B-Instruct
# export MODEL=meta-llama/Llama-3.1-8B-Instruct
export MODEL=bdsaglam/Qwen2.5-1.5B-Instruct-ragent-musique
export RETRIEVER=hybrid
export RETRIEVER_TOP_K=2
python scripts/ragent.py predict \
    --model $MODEL \
    --dataset-path bdsaglam/musique-mini \
    --dataset-name answerable \
    --dataset-split validation \
    --retriever $RETRIEVER \
    --retriever-top-k $RETRIEVER_TOP_K \
    --n-env-jobs 32 \
    --batch-size 32 \
    --out outputs/ragent/$MODEL/predictions-musique-mini-$RETRIEVER-$RETRIEVER_TOP_K.jsonl
```


## Rerank

```sh
curl 127.0.0.1:8930/rerank \
    -X POST \
    -d '{"query": "What is Deep Learning?", "texts": ["Neural networks are a type of machine learning model.", "Symbolic AI is a type of machine learning model."]}' \
    -H 'Content-Type: application/json'

curl http://localhost:8931/rerank \
    -X POST \
    -d '{"query": "What is Deep Learning?", "texts": ["Neural networks are a type of machine learning model.", "Symbolic AI is a type of machine learning model."]}' \
    -H 'Content-Type: application/json'

curl http://localhost:8931/rerank \
    -X POST \
    -d '{"query": "What is Deep Learning?", "texts": ["Neural networks are a type of machine learning model.", "Symbolic AI is a type of machine learning model."], "model": "tei"}' \
    -H 'Content-Type: application/json'
```

## Commands

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

python scripts/merge.py outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250427_095331 --out outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250427_095331-merged

## 2025-05-05

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
    --gradient-accumulation-steps 2 \
    --n-epochs 2 \
    --lora-r 256 \
    --lora-alpha 256 \
    2>&1 | tee tmp/logs/train-$(date +%s).log


accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --run-name 'Llama-3.1-8B-Instruct-ragent-grpo-20250505_172203' \
    --resume-from-checkpoint \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --n-epochs 4 \
    --lora-r 128 \
    --lora-alpha 256 \
    2>&1 | tee tmp/logs/train-$(date +%s).log


## 2025-05-08

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
    --gradient-accumulation-steps 2 \
    --n-epochs 2 \
    --lora-r 256 \
    --lora-alpha 256 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 3 \
    scripts/ragent.py train \
    --datasets 'bdsaglam/musique,answerable,train' \
    --model 'meta-llama/Llama-3.1-8B-Instruct' \
    --run-name 'Llama-3.1-8B-Instruct-ragent-grpo-20250508_095113' \
    --resume-from-checkpoint \
    --few-shot-prob 0.0 \
    --temperature 0.5 \
    --retriever 'hybrid-tei' \
    --retriever-top-k 1 \
    --n-env-jobs 32 \
    --batch-size 32 \
    --num-generations 8 \
    --gradient-accumulation-steps 2 \
    --n-epochs 2 \
    --lora-r 256 \
    --lora-alpha 256 \
    2>&1 | tee tmp/logs/train-$(date +%s).log

```sh
python scripts/merge.py \
    ./outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250508_213215 \
    --out outputs/Llama-3.1-8B-Instruct-ragent-grpo-20250508_213215-merged
```
