#!/bin/bash

set -e

start=2080
end=2080
increment=5000

upload_base_dir=/bb/llm/gaf51275/llama/converted-hf-checkpoint/Swallow-7b-VE-chat/imitation-1-and-2-lr_2e-5-minlr_2e-6-GB_64

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python tools/model-upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Swallow-7b-VE-instruct-v1.0-imitation-1-and-2-lr_2e-5-minlr_2e-6-GB_64-iter$(printf "%07d" $i)
done
