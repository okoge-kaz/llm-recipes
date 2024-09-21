#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:00:00
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.20.5
module load ylab/hpcx/2.18.1
module load ninja/1.11.1

# Set environment variables
source .env/bin/activate

# pip version up
pip install --upgrade pip
pip install --upgrade wheel cmake ninja packaging

# pip install requirements
pip install -r requirements.txt

# distirbuted training requirements
pip install mpi4py

# huggingface requirements
pip install huggingface_hub

# install flash-atten
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation
