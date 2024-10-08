#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:01:00:00
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

INFERENCE_MODEL_DIR=/bb/llm/gaf51275/2024/checkpoints/pytorch-to-hf/Llama-3.1-8B-Instruct/exp2-4/LR_1e-6_MINLR_1e-7_WD_0.1_GC_1/iter_0004000

python tools/inference/inference.py \
  --model-path $INFERENCE_MODEL_DIR \
  --tokenizer-path $INFERENCE_MODEL_DIR \
  --prompt "Please explain Credit Default Swaps." \
  --chat-template

python tools/inference/inference.py \
  --model-path $INFERENCE_MODEL_DIR \
  --tokenizer-path $INFERENCE_MODEL_DIR \
  --prompt "会社法について説明してください。" \
  --chat-template

python tools/inference/inference.py \
  --model-path $INFERENCE_MODEL_DIR \
  --tokenizer-path $INFERENCE_MODEL_DIR \
  --prompt "東京工業大学のキャンパスはどこにありますか？" \
  --chat-template

python tools/inference/inference.py \
  --model-path $INFERENCE_MODEL_DIR \
  --tokenizer-path $INFERENCE_MODEL_DIR \
  --prompt "1+4+8の答えはいくつでしょうか？" \
  --chat-template

python tools/inference/inference.py \
  --model-path $INFERENCE_MODEL_DIR \
  --tokenizer-path $INFERENCE_MODEL_DIR \
  --prompt "Pythonでデータ構造のUnionFindクラスを作成してください。" \
  --chat-template
