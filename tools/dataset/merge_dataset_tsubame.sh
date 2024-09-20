#!/bin/bash

set -e

# Control variables
INCLUDE_REDACTED=false
FILTERD_SCORE=0
NEXT_TOKEN_PERCENT=0
USE_OPEN_ASSISTANT=false
USE_ONLY_ENGLISH_OPEN_ASSISTANT=false
USE_ENGLISH_LMSYS=true
USE_MAGPIE_ULTRA_EN=true
USE_MAGPIE_ULTRA_JA=false
USE_GEMMA_MAGPIE=false
CUSTOM_OUTPUT_DIR=""

# Base output directory
BASE_OUTPUT_DIR="/gs/bs/tga-NII-LLM/datasets/raw/instruct/training"

if [ -z "$CUSTOM_OUTPUT_DIR" ]; then
  OUTPUT_DIR="$BASE_OUTPUT_DIR/exp2-filtered-$FILTERD_SCORE-next_token-$NEXT_TOKEN_PERCENT"
else
  OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
fi

if $INCLUDE_REDACTED; then
  OUTPUT_DIR="${OUTPUT_DIR}-redacted"
fi

if ! $USE_OPEN_ASSISTANT; then
  OUTPUT_DIR="${OUTPUT_DIR}-no-oasst"
elif $USE_ONLY_ENGLISH_OPEN_ASSISTANT; then
  OUTPUT_DIR="${OUTPUT_DIR}-en-oasst"
fi

if $USE_ENGLISH_LMSYS; then
  OUTPUT_DIR="${OUTPUT_DIR}-en-lmsys"
fi

if $USE_MAGPIE_ULTRA_en; then
  OUTPUT_DIR="${OUTPUT_DIR}-magpie-ultra-en"
fi

if $USE_MAGPIE_ULTRA_JA; then
  OUTPUT_DIR="${OUTPUT_DIR}-magpie-ultra-ja"
fi

if $USE_GEMMA_MAGPIE; then
  OUTPUT_DIR="${OUTPUT_DIR}-gemma-magpie"
fi

mkdir -p $OUTPUT_DIR

if $USE_OPEN_ASSISTANT; then
  if $USE_ONLY_ENGLISH_OPEN_ASSISTANT; then
    FILES=(
      "/gs/bs/tga-NII-LLM/datasets/raw/instruct/general/oasst2-top1-en-chat-sft/data/train.jsonl"
    )
  else
    FILES=(
      "/gs/bs/tga-NII-LLM/datasets/raw/instruct/general/oasst2-33k-ja/lm_filtered_split_1.jsonl"
      "/gs/bs/tga-NII-LLM/datasets/raw/instruct/general/oasst2-33k-ja/lm_filtered_split_2.jsonl"
      "/gs/bs/tga-NII-LLM/datasets/raw/instruct/general/oasst2-33k-ja/lm_filtered_split_3.jsonl"
      "/gs/bs/tga-NII-LLM/datasets/raw/instruct/general/oasst2-33k-ja/lm_filtered_split_4.jsonl"
      "/gs/bs/tga-NII-LLM/datasets/raw/instruct/general/oasst2-top1-en-chat-sft/data/train.jsonl"
    )
  fi

  MERGED_FILE=$OUTPUT_DIR/merged_oasst.jsonl

  for FILE in "${FILES[@]}"; do
    cat $FILE >> $MERGED_FILE
  done

  # filter
  python tools/dataset/fileter.py \
    --input_file $MERGED_FILE \
    --output_file $OUTPUT_DIR/train.jsonl \
    --threshold $FILTERD_SCORE

  rm $MERGED_FILE

  echo "Filtered open assistant data:"
  wc -l $OUTPUT_DIR/train.jsonl
else
  # Open Assistant データを使用しない場合は空のファイルを作成
  touch $OUTPUT_DIR/train.jsonl
  echo "Skipped Open Assistant data processing."
fi

# 日本語のLMSYSデータを常に使用
if $INCLUDE_REDACTED; then
  JA_LMSYS_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/lmsys-chat-1m/sft/lmsys-chat-1m-train.jsonl
else
  JA_LMSYS_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/lmsys-chat-1m/sft/lmsys-chat-1m-train-ja-no-redacted.jsonl
fi

cat $JA_LMSYS_FILE >> $OUTPUT_DIR/train.jsonl

# 英語のLMSYSデータを追加でオプションとして使用
if $USE_ENGLISH_LMSYS; then
  if $INCLUDE_REDACTED; then
    EN_LMSYS_FILE=/gs/bs/tga-NII-LLM/datasets/raw/instruct/lmsys-chat-1m/sft/lmsys-chat-1m-train-en.jsonl
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

INSTRUCTION_SAMPLES=$(wc -l $OUTPUT_DIR/train.jsonl | awk '{print $1}')
NEXT_TOKEN_SAMPLES=$(echo "($INSTRUCTION_SAMPLES / (1 - $NEXT_TOKEN_PERCENT)) * $NEXT_TOKEN_PERCENT / 1" | bc)

python tools/dataset/extract_jsonl.py \
  --input-path /gs/bs/tga-NII-LLM/datasets/raw/instruct/next-token/next-token-prediction_500k/format/merged.jsonl \
  --output-path $OUTPUT_DIR/next-token.jsonl \
  --num-lines $NEXT_TOKEN_SAMPLES \
  --seed 1234

echo "Next token data:"
wc -l $OUTPUT_DIR/next-token.jsonl

cat $OUTPUT_DIR/next-token.jsonl >> $OUTPUT_DIR/train.jsonl

echo "Total data:"
wc -l $OUTPUT_DIR/train.jsonl

rm $OUTPUT_DIR/next-token.jsonl

# indexing

python tools/pre-process/index_dataset.py \
  --data-file-path $OUTPUT_DIR/train.jsonl
