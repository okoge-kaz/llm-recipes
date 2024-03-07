#!/bin/bash

set -e

# swich virtual env
source .env/bin/activate

# fsdp
start=672
end=672
increment=5000

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/model/fujii/checkpoints/Swallow-7b/imitative-open-asst2-lr_2e-5-minlr_2e-6/${FORMATTED_ITERATION}/model.pt
  OUTPUT_PATH=/model/fujii/hf_checkpoints/swallow-7b/imitative-open-asst2/${FORMATTED_ITERATION}

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/model/fujii/hf_checkpoints/Swallow-7b-hf/

  python tools/checkpoint-convert/convert_ckpt.py \
    --model $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $OUTPUT_PATH \
    --sequence-length 4096
done
