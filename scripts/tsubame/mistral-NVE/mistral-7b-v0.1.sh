#!/bin/sh
#$ -cwd
#$ -l node_f=4
#$ -l h_rt=24:00:00
#$ -o outputs/mistral-7b-NVE/${JOB_ID}.out
#$ -e outputs/mistral-7b-NVE/${JOB_ID}.out
#$ -p -3

# priotiry: -5: normal, -4: high, -3: highest

# Load modules
module load cuda/12.1.0
module load nccl/2.20.5
module load openmpi/5.0.2-gcc
module load ninja/1.11.1
module load ~/modulefiles/cudnn/9.0.0

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
SEQ_LENGTH=4096
SLIDING_WINDOW_SIZE=4096
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=25000

# optimizer config
LR=2e-5
MIN_LR=6.6e-7
LR_WARMUP_STEPS=1000
LR_DECAY_STEPS=25000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL=/gs/bs/tge-gc24sp01/hf-checkpoints/Mistral-7B-v0.1/tokenizer.model
CHECKPOINT_DIR=/gs/bs/tge-gc24sp01/hf-checkpoints/Mistral-7B-v0.1
CHECKPOINT_SAVE_DIR="/gs/bs/tge-gc24sp01/checkpoints/Mistral-7b-v0.1-NVE/lr_${LR}-minlr_${MIN_LR}_warmup_${LR_WARMUP_STEPS}_sliding_window_${SLIDING_WINDOW_SIZE}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH=""

# ja okazaki lab cc
DATA_PATH="${DATA_PATH} /gs/bs/tge-gc24sp01/datasets/mistral_original_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_0_text_document"

# job name
JOB_NAME="Mistral-7b-NVE-t4-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none \
  -x PATH \
  python examples/finetuning.py \
  --seq-length ${SEQ_LENGTH} \
  --sliding-window-size ${SLIDING_WINDOW_SIZE} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --data-path ${DATA_PATH} \
  --split 949,50,1 \
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
  --adam-eps 1e-6 \
  --save-interval 500 \
  --eval-interval 100 \
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
  --wandb-entity "prj-jalm" \
  --wandb-project "T4-Mistral-7B-v0.1" \
  --wandb-name "${JOB_NAME}"
