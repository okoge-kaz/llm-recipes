#!/bin/bash

set -e

start=1284
end=1284
increment=5000

upload_base_dir=/bb/llm/gaf51275/llama/converted-hf-checkpoint/Swallow-7b-VE-chat/imitation-2-oasst2-top1-lr_2e-5-minlr_2e-6-GB_64

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python tools/model-upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Swallow-7b-VE-instruct-v1.0-imitation-2-oasst2-top1-lr_2e-5-minlr_2e-6-GB_64-iter$(printf "%07d" $i)
done
