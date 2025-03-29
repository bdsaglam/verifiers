#!/bin/sh

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='2' \
    -S devices='"0,1"' \
    -S run='1' 

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='1' \
    -S devices='"2,3"'\
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='3' \
    -S devices='"0,1"' \
    -S run='1' 

