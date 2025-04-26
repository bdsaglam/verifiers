#!/bin/sh

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
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
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

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
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
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
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
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
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='2' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-Coder-32B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.7' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-32B-Instruct' \
    -S model.temperature='0.3' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='Qwen/QwQ-32B' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='Qwen/Qwen2.5-14B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='3' \
    -S retriever.mode='all' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

