#!/bin/sh

dvc exp run --queue \
    -S model.path='meta-llama/Llama-3.1-8B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1900' \
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
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-2' \
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
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-400' \
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
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1900' \
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
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1600' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1600' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-2' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
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
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1600' \
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
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1900' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"2"'

dvc exp run --queue \
    -S model.path='meta-llama/Llama-3.1-8B-Instruct' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1600' \
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
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
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
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-2' \
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
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-400' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-2' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"1"'

dvc exp run --queue \
    -S model.path='meta-llama/Llama-3.1-8B-Instruct' \
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
    -S model.path='bdsaglam/Llama-3.1-8B-Instruct-ragent-grpo-musique-merged' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

dvc exp run --queue \
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-1900' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-400' \
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
    -S model.path='meta-llama/Llama-3.1-8B-Instruct' \
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
    -S model.path='/home/baris/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250421_000014-400' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S retriever.mode='new' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique-mini' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"3"'

