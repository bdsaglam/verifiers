#!/bin/sh

dvc exp run --queue \
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

