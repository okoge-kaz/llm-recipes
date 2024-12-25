#!/bin/bash

set -e

# Base output directory
BASE_OUTPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/instruct/training"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/v0.3"

MAGPIE_ULTRA_FILE=$OUTPUT_DIR/filtered-magpie-ultra-v0.1.jsonl
echo "magpie-ultra file: $MAGPIE_ULTRA_FILE"
wc -l $MAGPIE_ULTRA_FILE
cat $MAGPIE_ULTRA_FILE >> $OUTPUT_DIR/train.jsonl
echo "Added magpie-ultra data"

GEMMA_MAGPIE_FILE=$OUTPUT_DIR/gemma-magpie.jsonl
echo "gemma-magpie file: $GEMMA_MAGPIE_FILE"
wc -l $GEMMA_MAGPIE_FILE
cat $GEMMA_MAGPIE_FILE >> $OUTPUT_DIR/train.jsonl
echo "Added gemma-magpie data"

LMSYS_FILE=$OUTPUT_DIR/lmsys-chat-1m.jsonl
echo "lmsys file: $LMSYS_FILE"
wc -l $LMSYS_FILE
cat $LMSYS_FILE >> $OUTPUT_DIR/train.jsonl
echo "Added lmsys data"

echo "Total data:"
wc -l $OUTPUT_DIR/train.jsonl

# indexing

python tools/pre-process/index_dataset.py \
  --data-file-path $OUTPUT_DIR/train.jsonl
