#!/bin/bash
#$ -l rt_AF=1
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

CHECKPOINT_DIR=/bb/llm/gaf51275/checkpoints/llm-recipes/Llama-3.2-1B-Instruct/LR_4E-5_minLR_4E-6_WD_0.1_GC_1/
LATEST_ITERATION=$(cat ${CHECKPOINT_DIR}/latest_iteration.txt)

echo "LATEST_ITERATION=${LATEST_ITERATION}"

ITERATION=100000
echo > ${CHECKPOINT_DIR}/latest_iteration.txt

BASE_MODEL_CHECKPOINT=/bb/llm/gaf51275/hf-checkpoints/Llama-3.2-1B
TOKENIZER_DIR=/bb/llm/gaf51275/hf-checkpoints/Llama-3.2-1B
OUTPUT_DIR=/bb/llm/gaf51275/2024/checkpoints/pytorch-to-hf/Llama-3.2-1B/from-instruct
EXTRACTED_PATH=$(echo $CHECKPOINT_DIR | awk -F'/Llama-3.2-1B/' '{print $2}')
OUTPUT_DIR="${OUTPUT_DIR}${EXTRACTED_PATH}"

echo "convert ${CHECKPOINT_DIR} to ${OUTPUT_DIR}"
mkdir -p $OUTPUT_DIR

ITERATION=$ITERATION
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

echo $LATEST_ITERATION > ${CHECKPOINT_DIR}/latest_iteration.txt

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

EXP_NAME=$EXTRACTED_PATH
HF_REPO_NAME="tokyotech-llm/Llama-3.2-1B-from-instruct-LR_4E-5_minLR_4E-6_WD_0.1_GC_1-${FORMATTED_ITERATION}"

echo "upload ${OUTPUT_PATH} to ${HF_REPO_NAME}"

if ! upload_checkpoint "$OUTPUT_PATH" "$HF_REPO_NAME"; then
  echo "Skipping to next checkpoint after repeated failures for $HF_REPO_NAME"
fi
