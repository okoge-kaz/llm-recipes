#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=4:00:00
#$ -j y
#$ -o outputs/
#$ -cwd

set -e

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

# pip version up
pip install --upgrade pip

# pip install requirements
pip install -r requirements.txt

# distirbuted training requirements
pip install mpi4py

# huggingface requirements
pip install huggingface_hub

# install flash-atten
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation
