#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/tokenize/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module use /groups/gag51395/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.17/2.17.1-1
module load hpcx/2.12
module load gcc/11.4.0

# swich virtual env
source .env/bin/activate

DATASET_DIR=/bb/llm/gaf51275/datasets/raw
OUTPUT_DIR=/bb/llm/gaf51275/binarized/phi-3-default

mkdir -p ${OUTPUT_DIR}

# tokenize japanese wikipedia
python megatron_lm/tools/preprocess_data.py \
  --input ${DATASET_DIR}/ja_wiki_merged.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /bb/llm/gaf51275/hf-checkpoints/Phi-3-medium-4k-instruct/tokenizer.model \
  --append-eod \
  --workers 64
