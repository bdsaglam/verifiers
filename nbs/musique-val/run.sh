#!/bin/sh

dvc exp run --queue \
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-20250531_141657' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S repeat='1' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2,3"'

