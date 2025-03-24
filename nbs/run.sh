#!/bin/sh

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='3' \
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='2' \
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S retriever.name='lexical' \
    -S retriever.top_k='3' \
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S retriever.name='lexical' \
    -S retriever.top_k='2' \
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S retriever.name='lexical' \
    -S retriever.top_k='2' \
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='3' \
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S retriever.name='lexical' \
    -S retriever.top_k='1' \
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S retriever.name='lexical' \
    -S retriever.top_k='3' \
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='1' \
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='1' \
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='2' \
    -S run='1'

dvc exp run --queue \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S model.name='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-scaled' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S retriever.name='lexical' \
    -S retriever.top_k='1' \
    -S run='1'

