#!/bin/bash

OUTPUT_DIR=/bb/llm/gaf51275/datasets/raw/instruct/next-token/next-token-prediction_500k/format

mkdir -p $OUTPUT_DIR

FILES=$(find /bb/llm/gaf51275/datasets/raw/instruct/next-token/next-token-prediction_500k/format -name "*.jsonl")

MERGED_FILE=$OUTPUT_DIR/merged.jsonl

for FILE in "${FILES[@]}"; do
  cat $FILE >> $MERGED_FILE
done

wc -l $MERGED_FILE
