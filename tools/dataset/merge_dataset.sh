#!/bin/bash

INPUT_DIR=/bb/llm/gaf51275/datasets/raw/instruct/general/oasst2-33k-ja
OUTPUT_DIR=/bb/llm/gaf51275/datasets/raw/instruct/training/exp1-1

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
  --threshold 0

rm $MERGED_FILE
