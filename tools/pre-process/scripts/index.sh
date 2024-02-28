#!/bin/bash

source .env/bin/activate

INPUT_DIR=/bb/llm/gaf51275/llama/finetuning/datasets/training

# baseline
python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/baseline/train.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/baseline/val.jsonl

# baseline-imitation_2
python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/baseline-imitation_2/train.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/baseline-imitation_2/val.jsonl

# ichikara
python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/ichikara/train.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/ichikara/val.jsonl

# imitation_1_and_2
python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/imitation_1_and_2/train.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/imitation_1_and_2/val.jsonl

# imitation_2_oasst2_top1
python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/imitation_2_oasst2_top1/train.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/imitation_2_oasst2_top1/val.jsonl
