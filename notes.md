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
export CUDA_VISIBLE_DEVICES=0,1,2
accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 2 \
    scripts/ragent.py train \
    2>&1 | tee tmp/ragent-$(date +%s).log
```

```sh
export CUDA_VISIBLE_DEVICES=1,2,3
accelerate launch \
    --config-file configs/zero3.yaml \
    --num-processes 2 \
    scripts/ragent.py train \
    --model 'outputs/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    --few-shot-prob 1.0 \
    --retriever 'hybrid' \
    --retriever-top-k 1 \
    --n-env-jobs 1 \
    2>&1 | tee tmp/ragent-llama3-8b-round-2-$(date +%s).log
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
    --few-shot-prob 1.0 \
    --retriever 'hybrid' \
    --n-env-jobs 16 \
    --retriever-top-k 1 \
    --resume-from-checkpoint \
    2>&1 | tee tmp/ragent-llama3-8b-round-2-$(date +%s).log

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
