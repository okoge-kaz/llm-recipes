#!/bin/bash

# swich virtual env
source .env/bin/activate

DATASET_DIR=/home/kazuki/datasets/samples
OUTPUT_DIR=/home/kazuki/datasets/debug/yi-1.5

mkdir -p ${OUTPUT_DIR}

# tokenize japanese wikipedia
python megatron_lm/tools/preprocess_data.py \
  --input ${DATASET_DIR}/ja_wiki.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /home/kazuki/hf_checkpoints/Yi-1.5-9B/tokenizer.model \
  --append-eod \
  --workers 64
