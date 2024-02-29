#!/bin/bash

set -e

start=2430
end=2430
increment=5000

upload_base_dir=/bb/llm/gaf51275/llama/converted-hf-checkpoint/Swallow-7b-VE-chat/baseline-lr_2e-5-minlr_2e-6/

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python tools/model-upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Swallow-7b-VE-instruct-v1.0-baseline-lr_2e-5-minlr_2e-6-iter$(printf "%07d" $i)
done
