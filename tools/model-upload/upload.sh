#!/bin/bash

set -e

start=9250
end=9250
increment=2500

EXPERIMENT_NAME=exp1

upload_base_dir=/bb/llm/gaf51275/2024/checkpoints/Llama-3-8b-instruct-v0.2/${EXPERIMENT_NAME}/LR1.0E-5-MINLR1.0E-6-WD0.1

upload_checkpoint() {
  local upload_dir=$1
  local repo_name=$2
  local max_retries=5
  local retry_count=0

  while [ $retry_count -lt $max_retries ]; do
    if python scripts/abci/upload/upload.py \
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
  repo_name="tokyotech-llm/Llama-3-8b-instruct-v0.2-${EXPERIMENT_NAME}-LR1.0e-5-MINLR1.0E-6-iter$(printf "%07d" $i)"

  if ! upload_checkpoint "$upload_dir" "$repo_name"; then
    echo "Skipping to next checkpoint after repeated failures for $repo_name"
  fi
done
