#!/bin/bash

set -e

# Control variables
USE_ENGLISH_LMSYS=true
USE_SYNTH_EN_WO_TEMPLATE=true
USE_SYNTH_JA_WO_TEMPLATE=true
USE_MAGPIE_ULTRA_EN=true
USE_MAGPIE_ULTRA_JA=false
USE_GEMMA_MAGPIE=false

# Base output directory
BASE_OUTPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/instruct/training"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/exp2-12-3"

mkdir -p $OUTPUT_DIR
if [ -f $OUTPUT_DIR/train.jsonl ]; then
  rm $OUTPUT_DIR/train.jsonl
fi

# Japanese LMSYS
if $USE_SYNTH_JA_WO_TEMPLATE; then
  JA_LMSYS_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/lmsys-chat-1m/sft/lmsys-chat-1m-synth-ja-wo-pii-and-template-instructions-train.jsonl
else
  JA_LMSYS_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/lmsys-chat-1m/sft/lmsys-chat-1m-train-ja-no-redacted.jsonl
fi

cat $JA_LMSYS_FILE >> $OUTPUT_DIR/train.jsonl

# English LMSYS
if $USE_ENGLISH_LMSYS; then
  if $USE_SYNTH_EN_WO_TEMPLATE; then
    EN_LMSYS_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/lmsys-chat-1m/sft/lmsys-chat-1m-synth-en-wo-pii-and-template-instructions-train.jsonl
  else
    EN_LMSYS_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/lmsys-chat-1m/sft/lmsys-chat-1m-train-en-no-redacted.jsonl
  fi

  cat $EN_LMSYS_FILE >> $OUTPUT_DIR/train.jsonl
  echo "Added English LMSYS data"
fi

# Add magpie-ultra dataset processing
if $USE_MAGPIE_ULTRA_EN; then
  MAGPIE_ULTRA_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data/train_en.jsonl
  cat $MAGPIE_ULTRA_FILE >> $OUTPUT_DIR/train.jsonl
  echo "Added magpie-ultra en data"
fi

if $USE_MAGPIE_ULTRA_JA; then
  MAGPIE_ULTRA_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data/train_ja.jsonl
  cat $MAGPIE_ULTRA_FILE >> $OUTPUT_DIR/train.jsonl
  echo "Added magpie-ultra ja data"
fi

# Add gemma-magpie dataset processing
if $USE_GEMMA_MAGPIE; then
  GEMMA_MAGPIE_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/MAGPIE/gemma2-27b-it/format.jsonl
  cat $GEMMA_MAGPIE_FILE >> $OUTPUT_DIR/train.jsonl
  echo "Added gemma-magpie data"
fi

echo "Total data:"
wc -l $OUTPUT_DIR/train.jsonl

# indexing

python tools/pre-process/index_dataset.py \
  --data-file-path $OUTPUT_DIR/train.jsonl
