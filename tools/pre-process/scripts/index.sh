#!/bin/bash

source .env/bin/activate

INPUT_DIR=/gs/bs/tga-NII-LLM/datasets/raw/instruct/synthetic/general/Synthetic-JP-Conversations-Magpie-Nemotron-4-10k

# baseline
python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/Synthetic-JP-Conversations-Magpie-Nemotron-4-10k.jsonl
