#!/bin/bash

source .env/bin/activate

INPUT_DIR=/bb/llm/gaf51275/llama/finetuning/datasets/training

# dolly-oasst2-top1-imitation-2-3
python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/dolly-oasst2-top1-imitation-2-3/train.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/dolly-oasst2-top1-imitation-2-3/val.jsonl

# oasst2-top1-imitation-2-3
python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/oasst2-top1-imitation-2-3/train.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/oasst2-top1-imitation-2-3/val.jsonl
