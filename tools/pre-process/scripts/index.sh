#!/bin/bash

source .env/bin/activate

INPUT_DIR=/bb/llm/gaf51275/datasets/raw/instruct/training/exp1-1

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/train.jsonl

INPUT_DIR=/bb/llm/gaf51275/datasets/raw/instruct/training/exp1-3

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/train.jsonl

INPUT_DIR=/bb/llm/gaf51275/datasets/raw/instruct/training/exp1-4

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/train.jsonl
