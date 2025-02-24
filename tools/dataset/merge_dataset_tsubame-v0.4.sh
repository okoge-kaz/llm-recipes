#!/bin/bash

set -e

# Base output directory
BASE_OUTPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/instruct/training"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/v0.4"

MAGPIE_ULTRA_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/training/v0.3/filtered-magpie-ultra-v0.1.jsonl
echo "magpie-ultra file: $MAGPIE_ULTRA_FILE"
wc -l $MAGPIE_ULTRA_FILE
cat $MAGPIE_ULTRA_FILE >> $OUTPUT_DIR/train_general.jsonl
echo "Added magpie-ultra data"

GEMMA_MAGPIE_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/training/v0.3/gemma-magpie.jsonl
echo "gemma-magpie file: $GEMMA_MAGPIE_FILE"
wc -l $GEMMA_MAGPIE_FILE
cat $GEMMA_MAGPIE_FILE >> $OUTPUT_DIR/train_general.jsonl
echo "Added gemma-magpie data"

LMSYS_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/training/v0.3/lmsys-chat-1m.jsonl
echo "lmsys file: $LMSYS_FILE"
wc -l $LMSYS_FILE
cat $LMSYS_FILE >> $OUTPUT_DIR/train_general.jsonl
echo "Added lmsys data"

EN_LMSYS_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/lmsys-chat-1m/sft/lmsys-chat-1m-synth-en-wo-pii-and-template-instructions-train.jsonl
echo "en lmsys file: $EN_LMSYS_FILE"
wc -l $EN_LMSYS_FILE
cat $EN_LMSYS_FILE >> $OUTPUT_DIR/train_general.jsonl
echo "Added en lmsys data"

MAGPIE_ULTRA_EN_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data/train_en.jsonl
echo "magpie-ultra en file: $MAGPIE_ULTRA_EN_FILE"
wc -l $MAGPIE_ULTRA_EN_FILE
cat $MAGPIE_ULTRA_EN_FILE >> $OUTPUT_DIR/train_general.jsonl
echo "Added magpie-ultra en data"

echo "Total data:"
wc -l $OUTPUT_DIR/train_general.jsonl

# indexing

python tools/pre-process/index_dataset.py \
  --data-file-path $OUTPUT_DIR/train_general.jsonl
