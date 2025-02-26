#!/bin/bash

set -e

start=5224
end=5224
increment=5000

upload_base_dir=/gs/bs/tga-NII-LLM/checkpoints/fsdp-to-hf/Llama-3.1-8B-Instruct-v0.4/exp1

upload_checkpoint() {
  local upload_dir=$1
  local repo_name=$2
  local max_retries=5
  local retry_count=0

  while [ $retry_count -lt $max_retries ]; do
    if python tools/model-upload/upload.py \
        --ckpt-path "$upload_dir" \
        --repo-name "$repo_name"; then
        echo "Successfully uploaded $repo_name"
        return 0
    else
        echo "Upload failed for $repo_name. Retrying..."
        ((retry_count++))
        sleep 5
    fi
  done

  echo "Failed to upload $repo_name after $max_retries attempts"
  return 1
}

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)
  repo_name="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.4-exp1-iter_$(printf "%07d" $i)"

  if ! upload_checkpoint "$upload_dir" "$repo_name"; then
    echo "Skipping to next checkpoint after repeated failures for $repo_name"
  fi
done
