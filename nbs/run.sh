#!/bin/sh

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0,1"'

