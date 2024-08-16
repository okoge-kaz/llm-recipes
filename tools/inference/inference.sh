#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=0:1:00:00
#$ -o outputs/inference/$JOB_ID.log
#$ -e outputs/inference/$JOB_ID.log
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

INFERENCE_MODEL_DIR=/gs/bs/tga-NII-LLM/checkpoints/fsdp-hf/Llama-3-8B-Instruct-v0.2/LR_1e-5_MINLR_1e-6_WD_0.1_GC_1-dist-ckpt/iter_0000078

python tools/inference/inference.py \
  --model-path $INFERENCE_MODEL_DIR \
  --tokenizer-path $INFERENCE_MODEL_DIR \
  --prompt "Tokyo is the capital of Japan."

python tools/inference/inference.py \
  --model-path $INFERENCE_MODEL_DIR \
  --tokenizer-path $INFERENCE_MODEL_DIR \
  --prompt "東京工業大学のキャンパスは"
