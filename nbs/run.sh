#!/bin/sh

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled-merged' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S retriever.name='semantic' \
    -S retriever.top_k='1' \
    -S run='1' \
    -S devices='0'

