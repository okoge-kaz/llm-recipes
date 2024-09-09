#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/convert/ckpt/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module use /bb/llm/gaf51275/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.20.5
module load hpcx/2.12
module load gcc/11.4.0

set -e
export HF_HOME="/groups/gag51395/.cache/huggigface"

# swich virtual env
source .env/bin/activate

CHECKPOINT_DIR=/bb/llm/gaf51275/2024/checkpoints/Llama-3.1-8B-Instruct/exp1-1/LR_1e-5_MINLR_1e-6_WD_0.1_GC_1
LATEST_ITERATION=$(cat ${CHECKPOINT_DIR}/latest_iteration.txt)

echo "LATEST_ITERATION=${LATEST_ITERATION}"

BASE_MODEL_CHECKPOINT=/bb/llm/gaf51275/hf-checkpoints/Meta-Llama-3.1-8B-Instruct
TOKENIZER_DIR=/groups/gag51395/hf-checkpoints/Meta-Llama-3-8B-Instruct
OUTPUT_DIR=/bb/llm/gaf51275/2024/checkpoints/pytorch-to-hf/Llama-3.1-8B-Instruct/
EXTRACTED_PATH=$(echo $CHECKPOINT_DIR | awk -F'/Llama-3.1-8B-Instruct/' '{print $2}')
OUTPUT_DIR="${OUTPUT_DIR}${EXTRACTED_PATH}"

echo "convert ${CHECKPOINT_DIR} to ${OUTPUT_DIR}"
mkdir -p $OUTPUT_DIR

ITERATION=$LATEST_ITERATION
FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

CHECK_POINT_PATH=${CHECKPOINT_DIR}/${FORMATTED_ITERATION}/model.pt
OUTPUT_PATH=${OUTPUT_DIR}/${FORMATTED_ITERATION}

echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

mkdir -p $OUTPUT_PATH

# convert
python tools/checkpoint-convert/convert_ckpt.py \
  --hf-base-model-checkpoint-path $BASE_MODEL_CHECKPOINT \
  --hf-tokenizer-path $TOKENIZER_DIR \
  --pytorch-model-checkpoint-path $CHECK_POINT_PATH \
  --out $OUTPUT_PATH \
  --sequence-length 8192

# upload
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

EXP_NAME=$(echo $EXTRACTED_PATH | sed 's/\//-/g')
HF_REPO_NAME="tokyotech-llm/Llama-3.1-8B-Instruct-${EXP_NAME}-${FORMATTED_ITERATION}"

echo "upload ${OUTPUT_PATH} to ${HF_REPO_NAME}"

if ! upload_checkpoint "$OUTPUT_PATH" "$HF_REPO_NAME"; then
  echo "Skipping to next checkpoint after repeated failures for $HF_REPO_NAME"
fi
