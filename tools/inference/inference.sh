#!/bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/inference/
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

# swich virtual env
source .env/bin/activate

python tools/inference/inference.py \
  --model-path /bb/llm/gaf51275/llama/converted-hf-checkpoint/mistral-7B-VE/okazaki-cc/iter_0004000 \
  --tokenizer-path /bb/llm/gaf51275/llama/converted-hf-checkpoint/mistral-7B-VE/okazaki-cc/iter_0004000 \
  --prompt "Tokyo is the capital of Japan."

python tools/inference/inference.py \
  --model-path /bb/llm/gaf51275/llama/converted-hf-checkpoint/mistral-7B-VE/okazaki-cc/iter_0004000 \
  --tokenizer-path /bb/llm/gaf51275/llama/converted-hf-checkpoint/mistral-7B-VE/okazaki-cc/iter_0004000 \
  --prompt "東京工業大学のキャンパスは"
