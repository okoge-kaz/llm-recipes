#!/bin/bash

set -e

start=20000
end=20000
increment=5000

upload_base_dir=/bb/llm/gaf51275/llama/converted-hf-checkpoint/mistral-7B-VE/the-vault-lr_2e-5-minlr_6.6e-7

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python tools/model-upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Mistral-7B-VE-the-vault-iter$(printf "%07d" $i)
done
