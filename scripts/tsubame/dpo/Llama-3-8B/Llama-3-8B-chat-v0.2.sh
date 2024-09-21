#!/bin/sh
#$ -cwd
#$ -l node_f=4
#$ -l h_rt=1:00:00:00
#$ -o outputs/Llama-3-8b-dpo/$JOB_ID.log
#$ -e outputs/Llama-3-8b-dpo/$JOB_ID.log
#$ -p -5

# module load
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# swich virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

# training config
SEQ_LENGTH=8192
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=128

# optimizer config
LR=1e-5
MIN_LR=1e-6
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint
TOKENIZER_DIR=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3-8B-Instruct
CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/swallow-hf/Llama-3-Swallow-8B-v0.1
CHECKPOINT_SAVE_DIR="/gs/bs/tga-NII-LLM/checkpoints/Llama-3-8B-chat-v0.2/LR_${LR}_MINLR_${MIN_LR}_WD_${WEIGHT_DECAY}_GC_${GRAD_CLIP}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# dataset
DATASET_DIR=/gs/bs/tga-NII-LLM/datasets/raw/dpo/hh-rlhf-12k-ja

TRAIN_DATA_PATH=${DATASET_DIR}/converted.jsonl
VALID_DATA_PATH=${DATASET_DIR}/converted.jsonl

# job name
JOB_NAME="Llama-3-8B-dpo-v0.2-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x LD_LIBRARY_PATH \
  -x PATH \
  -bind-to none \
  python train_llm.py \
  --seq-length ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --hf-transformer-model-dir ${TOKENIZER_DIR} \
  --dpo-train-data-path ${TRAIN_DATA_PATH} \
  --dpo-valid-data-path ${VALID_DATA_PATH} \
  --epoch 1 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
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
  --direct-preference-optimization \
  --dpo-beta 0.1 \
  --dpo-label-smoothing 0.0 \
  --save-sampler-state \
  --use-mpi \
  --wandb-entity "prj-jalm" \
  --wandb-project "Llama-3-8B-chat-v0.2" \
  --wandb-name "${JOB_NAME}"
