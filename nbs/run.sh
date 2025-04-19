#!/bin/sh

dvc exp run --queue \
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-2' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='all-metrics/1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0,2"'

dvc exp run --queue \
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-2' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='all-metrics/1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1,3"'

