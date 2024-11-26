#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:1:00:00
#$ -o outputs/inference/$JOB_ID.log
#$ -e outputs/inference/$JOB_ID.log
#$ -p -5

# module load
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

set -e

# swich virtual env
source .env/bin/activate

INFERENCE_MODEL_DIR=/gs/bs/tga-NII-LLM/checkpoints/megatron-to-hf/Llama-3.1-8b-v0.9/tp2-pp1-ct1/iter_0000100

python tools/inference/inference.py \
  --model-path $INFERENCE_MODEL_DIR \
  --tokenizer-path $INFERENCE_MODEL_DIR \
  --prompt "Tokyo is the capital of Japan."

python tools/inference/inference.py \
  --model-path $INFERENCE_MODEL_DIR \
  --tokenizer-path $INFERENCE_MODEL_DIR \
  --prompt "東京工業大学のキャンパスは"
