#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=0:05:00:00
#$ -o outputs/convert/$JOB_ID.log
#$ -e outputs/convert/$JOB_ID.log
#$ -p -5

# module load
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

set -e
export HF_HOME="/gs/bs/tga-NII-LLM/.cache"

# swich virtual env
source .env/bin/activate

CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/Llama-3.1-8B-Instruct/exp2-12-2/LR_2.5e-5_MINLR_2.5e-6_WD_0.1_GC_1
LATEST_ITERATION=$(cat ${CHECKPOINT_DIR}/latest_iteration.txt)

echo "LATEST_ITERATION=${LATEST_ITERATION}"

BASE_MODEL_CHECKPOINT=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-8B-Instruct
TOKENIZER_DIR=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3-8B-Instruct-pad-token
OUTPUT_DIR=/gs/bs/tgh-24IDU/checkpoints/pytorch-to-hf/Llama-3.1-8B-Instruct/
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
