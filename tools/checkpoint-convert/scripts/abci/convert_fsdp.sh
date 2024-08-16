#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:1:00:00
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

# swich virtual env
source .env/bin/activate

start=78
end=78
increment=1000

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/gs/bs/tga-NII-LLM/checkpoints/Llama-3-8B-Instruct-v0.2/LR_1e-5_MINLR_1e-6_WD_0.1_GC_1-dist-ckpt/${FORMATTED_ITERATION}
  OUTPUT_PATH=/gs/bs/tga-NII-LLM/checkpoints/fsdp-hf/Llama-3-8B-Instruct-v0.2/LR_1e-5_MINLR_1e-6_WD_0.1_GC_1-dist-ckpt/${FORMATTED_ITERATION}

  echo "convert FSDP ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3-8B-Instruct

  python tools/checkpoint-convert/convert_fsdp.py \
  --hf-base-model-path $BASE_MODEL_CHECKPOINT \
  --tokenizer-path $BASE_MODEL_CHECKPOINT \
  --fsdp-checkpoint-path $CHECK_POINT_PATH \
  --checkpoint-output-path $OUTPUT_PATH \
  --sequence-length 8192
done
