#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:00:00
#$ -p -5

# priotiry: -5: normal, -4: high, -3: highest

# Load modules
module load cuda/12.1.0
module load nccl/2.20.5
module load openmpi/5.0.2-gcc
module load ninja/1.11.1
module load ~/modulefiles/cudnn/9.0.0

# Set environment variables
source .env/bin/activate

pip install --upgrade pip

# Install packages
pip install -r requirements.txt

# flash attn
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation
