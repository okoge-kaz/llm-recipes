#!/bin/bash

OUTPUT_DIR=/bb/llm/gaf51275/datasets/raw/instruct/MAGPIE/gemma2-27b-it

mkdir -p $OUTPUT_DIR
rm $OUTPUT_DIR/merged.jsonl

FILES=$(find /bb/llm/gaf51275/datasets/raw/instruct/MAGPIE/gemma2-27b-it -name "*.jsonl")

MERGED_FILE=$OUTPUT_DIR/merged.jsonl

for FILE in "${FILES[@]}"; do
  cat $FILE >> $MERGED_FILE
done

wc -l $MERGED_FILE
