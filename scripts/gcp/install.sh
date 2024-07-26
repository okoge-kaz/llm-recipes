#!/bin/bash
#SBATCH --job-name=install
#SBATCH --partition=a3
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/install/%x-%j.out
#SBATCH --error=outputs/install/%x-%j.out

set -e

# module load
module load turing/cuda/12.1
module load turing/cudnn/8.9.7
module load turing/nccl/2.20.5
module load turing/hpcx/2.17.1

# swich virtual env
source .env/bin/activate

# pip version up
pip install --upgrade pip

# pip install requirements
pip install -r requirements.txt
pip install ninja packaging wheel

# distirbuted training requirements
pip install mpi4py

# huggingface requirements
pip install huggingface_hub

# install transformer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.6
pip uninstall flash-attn

# install flash-atten
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.4.2
pip install -e .
