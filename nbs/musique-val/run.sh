#!/bin/sh

dvc exp run --queue \
    -S agent='ragent' \
    -S model.path='Qwen/Qwen2.5-7B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S repeat='5' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S agent='ragent' \
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S repeat='5' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S agent='ragent' \
    -S model.path='Qwen/Qwen2.5-7B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S repeat='1' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S agent='ragent' \
    -S model.path='meta-llama/Llama-3.1-8B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S repeat='5' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S agent='ragent' \
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S repeat='1' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S agent='ragent' \
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809-merged' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S repeat='5' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S agent='ragent' \
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-20250520_080809-merged' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S repeat='1' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

