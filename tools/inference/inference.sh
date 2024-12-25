#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:1:00:00
#$ -o outputs/inference/$JOB_ID.log
#$ -e outputs/inference/$JOB_ID.log
#$ -p -3

# module load
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

set -e

# switch virtual env
source .env/bin/activate

INFERENCE_MODEL_DIR=/gs/bs/tga-NII-LLM/checkpoints/fsdp-to-hf/Llama-3.1-70B-Instruct-v0.3/LR_1.75e-5_MINLR_1.75e-6_WD_0.1_GC_1/iter_0002994

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
