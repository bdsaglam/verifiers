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
accelerate launch --config-file configs/zero3.yaml --num-processes 2 examples/ragent_train.py  2>&1 | tee tmp/ragent-qwen-musique-grpo.log
```

### Publish manually
```sh
huggingface-cli upload --repo-type model \
    ragent-Qwen2.5-1.5B-Instruct-musique-grpo \
    ./outputs/ragent-Qwen2.5-1.5B-Instruct-musique-grpo
```


### Resume training

```sh
accelerate launch --config-file configs/zero3.yaml --num-processes 2 examples/ragent_train.py \
    --model 'outputs/ragent-Qwen2.5-1.5B-Instruct-musique-grpo' \
    2>&1 | tee tmp/ragent-qwen-musique-grpo-resume.log

```

## Predict

```sh
python scripts/merge.py \
    outputs/ragent-Qwen2.5-1.5B-Instruct-musique-grpo/checkpoint-1000 \
    --out outputs/ragent-Qwen2.5-1.5B-Instruct-musique-grpo-merged
```

```sh
python examples/ragent_predict.py \
    --model outputs/ragent-Qwen2.5-1.5B-Instruct-musique-grpo-merged \
    --dataset-path bdsaglam/musique-mini \
    --dataset-name answerable \
    --dataset-split validation \
    --report-to none
```
