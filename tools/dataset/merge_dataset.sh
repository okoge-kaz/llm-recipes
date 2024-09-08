#!/bin/bash

set -e

INCLUDE_REDACTED=true
FILTERD_SCORE=7
NEXT_TOKEN_PERCENT=0.25

OUTPUT_DIR=/bb/llm/gaf51275/datasets/raw/instruct/training/exp2-filtered-$FILTERD_SCORE-next_token-$NEXT_TOKEN_PERCENT

if $INCLUDE_REDACTED; then
  OUTPUT_DIR=$OUTPUT_DIR-redacted
fi

mkdir -p $OUTPUT_DIR

FILES=(
  "/bb/llm/gaf51275/datasets/raw/instruct/general/oasst2-33k-ja/lm_filtered_split_1.jsonl"
  "/bb/llm/gaf51275/datasets/raw/instruct/general/oasst2-33k-ja/lm_filtered_split_2.jsonl"
  "/bb/llm/gaf51275/datasets/raw/instruct/general/oasst2-33k-ja/lm_filtered_split_3.jsonl"
  "/bb/llm/gaf51275/datasets/raw/instruct/general/oasst2-33k-ja/lm_filtered_split_4.jsonl"
  "/bb/llm/gaf51275/datasets/raw/instruct/general/oasst2-top1-en-chat-sft/data/lm_scored.jsonl"
)

MERGED_FILE=$OUTPUT_DIR/merged.jsonl

for FILE in "${FILES[@]}"; do
  cat $FILE >> $MERGED_FILE
done

# fileter
python tools/dataset/fileter.py \
  --input_file $MERGED_FILE \
  --output_file $OUTPUT_DIR/train.jsonl \
  --threshold $FILTERD_SCORE

rm $MERGED_FILE

echo "Filtered open assistant data:"
wc -l $OUTPUT_DIR/train.jsonl

if $INCLUDE_REDACTED; then
  LMSYS_FILE=/bb/llm/gaf51275/datasets/raw/instruct/lmsys-chat-1m/sft/lmsys-chat-1m-train.jsonl
else
  LMSYS_FILE=/bb/llm/gaf51275/datasets/raw/instruct/lmsys-chat-1m/sft/lmsys-chat-1m-train-no-redacted.jsonl
fi

cat $LMSYS_FILE >> $OUTPUT_DIR/train.jsonl

INSTRUCTION_SAMPLES=$(wc -l $OUTPUT_DIR/train.jsonl | awk '{print $1}')
NEXT_TOKEN_SAMPLES=$(echo "$INSTRUCTION_SAMPLES * $NEXT_TOKEN_PERCENT / 1" | bc)

python tools/dataset/extract_jsonl.py \
  --input-path /bb/llm/gaf51275/datasets/raw/instruct/next-token/next-token-prediction_500k/format/merged.jsonl \
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
