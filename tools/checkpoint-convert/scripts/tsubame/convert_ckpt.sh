#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:03:00:00
#$ -o outputs/convert/$JOB_ID.log
#$ -e outputs/convert/$JOB_ID.log
#$ -p -3

# module load
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

start=7812
end=7812
increment=5000

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/gs/bs/tga-NII-LLM/checkpoints/Llama-3.1-8B-Instruct-v0.4/exp3-stage1/LR_2.5e-5_MINLR_2.5e-6_WD_0.1_GC_1/${FORMATTED_ITERATION}/model.pt
  OUTPUT_PATH=/gs/bs/tga-NII-LLM/checkpoints/fsdp-to-hf/Llama-3.1-8B-Instruct-v0.4/exp3-stage1/${FORMATTED_ITERATION}

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3-8B-Instruct-pad-token

  python tools/checkpoint-convert/convert_ckpt.py \
    --hf-base-model-checkpoint-path $BASE_MODEL_CHECKPOINT \
    --hf-tokenizer-path $BASE_MODEL_CHECKPOINT \
    --pytorch-model-checkpoint-path $CHECK_POINT_PATH \
    --out $OUTPUT_PATH \
    --sequence-length 131072
done
