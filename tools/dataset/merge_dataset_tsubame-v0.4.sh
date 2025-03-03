#!/bin/bash

set -e

# Base output directory
BASE_OUTPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/instruct/training"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/v0.4-70B"

mkdir -p $OUTPUT_DIR

MAGPIE_ULTRA_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/training/v0.4.1/filtered-magpie-ultra-v0.1-swallow-train-ja-formatted.jsonl
echo "magpie-ultra file: $MAGPIE_ULTRA_FILE"
wc -l $MAGPIE_ULTRA_FILE
cat $MAGPIE_ULTRA_FILE >> $OUTPUT_DIR/train_general.jsonl
echo "Added magpie-ultra data"

GEMMA_MAGPIE_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/training/v0.4.1/gemma-magpie-swallow-train-greater-7-formatted.jsonl
echo "gemma-magpie file: $GEMMA_MAGPIE_FILE"
wc -l $GEMMA_MAGPIE_FILE
cat $GEMMA_MAGPIE_FILE >> $OUTPUT_DIR/train_general.jsonl
echo "Added gemma-magpie data"

LMSYS_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/training/v0.4.1/lmsys-chat-1m-synth-ja-gemma2-2turn-wo-pii-and-template-instructions-sft.jsonl
echo "lmsys file: $LMSYS_FILE"
wc -l $LMSYS_FILE
cat $LMSYS_FILE >> $OUTPUT_DIR/train_general.jsonl
echo "Added lmsys data"

echo "Total data:"
wc -l $OUTPUT_DIR/train_general.jsonl

# indexing

python tools/pre-process/index_dataset.py \
  --data-file-path $OUTPUT_DIR/train_general.jsonl
