#!/bin/bash
#SBATCH --job-name=llama-3.1-8b
#SBATCH --partition=h100
#SBATCH --time=0-01:00:00
#SBATCH --nodes 8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/Llama-3.1-8b/%x-%j.out
#SBATCH --error=outputs/Llama-3.1-8b/%x-%j.out

set -e

module load gc1/cuda/12.1
module load gc1/cudnn/9.2.0
module load gc1/nccl/2.20.5
module load gc1/hpcx/2.18.1

source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="h100"

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# training config
SEQ_LENGTH=8192
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=512
TRAIN_STEPS=25000

# optimizer config
LR=2.5E-5
MIN_LR=2.5E-6
LR_WARMUP_STEPS=1000
LR_DECAY_STEPS=25000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL=/home/kazuki_fujii/hf-checkpoints/Meta-Llama-3.1-8B/tokenizer.json
CHECKPOINT_DIR=/home/kazuki_fujii/hf-checkpoints/Meta-Llama-3.1-8B
CHECKPOINT_SAVE_DIR="/home/kazuki_fujii/checkpoints/Llama-3.1-8b/llm-recipes/LR_${LR}-minLR_${MIN_LR}-WD_${WEIGHT_DECAY}-GC_${GRAD_CLIP}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH=""

# ja wikipedia
DATA_PATH="${DATA_PATH} 1689848183 /home/kazuki_fujii/datasets/pretrain/binarized/llama3/ja_wiki_text_document"

# job name
JOB_NAME="Llama-3.1-8b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

TENSORBOARD_DIR="${CHECKPOINT_SAVE_DIR}/tensorboard/${NUM_NODES}-nodes"
mkdir -p ${TENSORBOARD_DIR}

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none \
  -x NCCL_IB_TIMEOUT=22 \
  -x LD_LIBRARY_PATH \
  -x PATH \
  python train_llm.py \
  --seq-length ${SEQ_LENGTH} \
  --sliding-window-size ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama3Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --data-path ${DATA_PATH} \
  --split 990,10,0 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --lr-decay-iters ${LR_DECAY_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip-norm ${GRAD_CLIP} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --save-interval 500 \
  --eval-interval 500 \
  --eval-iters 10 \
  --bf16 \
  --mixed-precision \
  --base-model ${CHECKPOINT_DIR} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --low-cpu-fsdp \
  --sharding-strategy FULL_SHARD \
  --checkpoint-type LOCAL_STATE_DICT \
  --fsdp-activation-checkpointing \
  --use-mpi \
  --torch-profile \
  --torch-profile-active 2 \
  --torch-profile-record-shapes \
  --torch-profile-profile-memory \
  --torch-profile-with-stack \
  --torch-profile-with-flops \
  --torch-profile-with-modules \
  --tensorboard-dir ${TENSORBOARD_DIR} \
  --wandb-entity "okoge" \
  --wandb-project "llm-recipes-gaggle" \
  --wandb-name "${JOB_NAME}"
