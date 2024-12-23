#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:00:00
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

pip install --upgrade pip
pip install --upgrade wheel cmake ninja packaging

# install nvidia pytorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install packages
pip install -r requirements.txt

export MAX_JOBS=8

pip install flash-attn --no-build-isolation
