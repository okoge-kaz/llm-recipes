#!/bin/bash

# swich virtual env
source .env/bin/activate

DATASET_DIR=/home/ext_kazuki_fujii_rio_gsic_titech/datasets/samples
OUTPUT_DIR=/home/ext_kazuki_fujii_rio_gsic_titech/datasets/debug/Codestral-22B-v0.1

mkdir -p ${OUTPUT_DIR}

# tokenize japanese wikipedia
python megatron_lm/tools/preprocess_data.py \
  --input ${DATASET_DIR}/ja_wiki.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /home/ext_kazuki_fujii_rio_gsic_titech/hf_checkpoints/Codestral-22B-v0.1/tokenizer.model \
  --append-eod \
  --workers 64
