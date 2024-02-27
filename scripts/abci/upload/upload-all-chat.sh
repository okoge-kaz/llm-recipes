#!/bin/bash

set -e

start=10000
end=12000
increment=1000

upload_base_dir=/bb/gaf51217/fujii/hf_checkpoints/kotoba/llama-2-13b-chat-filtered-gbs_16

# for ループで指定された範囲と増分を使用
for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/kotoba/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name kotoba-tech/Llama2-13b-chat-gbs16-no-title-iter$(printf "%07d" $i)
done
