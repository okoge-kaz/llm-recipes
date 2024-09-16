#!/bin/bash
#SBATCH --job-name=install
#SBATCH --partition=h100
#SBATCH --time=0-01:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/install/%x-%j.out
#SBATCH --error=outputs/install/%x-%j.out

set -e

module load gc1/cuda/12.1
module load gc1/cudnn/9.2.0
module load gc1/nccl/2.20.5
module load gc1/hpcx/2.18.1

source .env/bin/activate

# pip install
pip install --upgrade pip
pip install --upgrade wheel cmake ninja

# pip install requirements
pip install -r requirements.txt

# distirbuted training requirements
pip install mpi4py

# huggingface requirements
pip install huggingface_hub

# install flash-atten
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation
