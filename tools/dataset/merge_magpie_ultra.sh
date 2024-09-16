#!/bin/bash

OUTPUT_DIR=/bb/llm/gaf51275/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data

mkdir -p $OUTPUT_DIR
rm $OUTPUT_DIR/merged.jsonl

FILES=(
  "/bb/llm/gaf51275/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data/lm_train-00000-of-00002_1.jsonl"
  "/bb/llm/gaf51275/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data/lm_train-00000-of-00002_2.jsonl"
  "/bb/llm/gaf51275/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data/lm_train-00000-of-00002_3.jsonl"
  "/bb/llm/gaf51275/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data/lm_train-00001-of-00002_1.jsonl"
  "/bb/llm/gaf51275/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data/lm_train-00001-of-00002_2.jsonl"
  "/bb/llm/gaf51275/datasets/raw/instruct/synthetic/magpie-ultra-v0.1/data/lm_train-00001-of-00002_3.jsonl"
)

MERGED_FILE=$OUTPUT_DIR/merged.jsonl

for FILE in "${FILES[@]}"; do
  cat $FILE >> $MERGED_FILE
done

wc -l $MERGED_FILE
