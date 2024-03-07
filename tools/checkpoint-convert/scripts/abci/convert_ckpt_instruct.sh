#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/convert/ckpt/
#$ -cwd
# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

set -e

# swich virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

start=578
end=578
increment=5000

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/bb/llm/gaf51275/llama/checkpoints/Swallow-70b-VE-chat/oasst2-top1-imitation-2-3-lr_1e-5-minlr_1e-6-GB_256/${FORMATTED_ITERATION}/model.pt
  OUTPUT_PATH=/bb/llm/gaf51275/llama/converted-hf-checkpoint/Swallow-70b-VE-chat/oasst2-top1-imitation-2-3-lr_1e-5-minlr_1e-6-GB_256/${FORMATTED_ITERATION}

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/bb/llm/gaf51275/llama/huggingface-checkpoint/Swallow-70b-hf

  python tools/checkpoint-convert/convert_ckpt.py \
    --model $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $OUTPUT_PATH \
    --sequence-length 4096
done
