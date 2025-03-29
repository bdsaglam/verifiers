#!/bin/sh

dvc exp run --queue \
    -S model.path='meta-llama/Llama-3.1-8B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S retriever.name='hybrid' \
    -S retriever.top_k='1' \
    -S run='1' \
    -S dataset.path='bdsaglam/hotpotqa-distractor' \
    -S dataset.name='default' \
    -S dataset.split='validation[:300]' \
    -S devices='"0"'

