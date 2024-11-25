#!/bin/sh
#PBS -q rt_HF
#PBS -N llama-3.1-8b
#PBS -l select=2:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/all_reduce_bench
#PBS -P gcg51558

cd $PBS_O_WORKDIR
mkdir -p outputs/Llama-3.1-8B

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /groups/gag51395/modules/modulefiles

module load cuda/12.4
module load cudnn/9.1.1
module load nccl/2.21.5
module load hpcx/2.18.1

source .env/bin/activate

# distributed settings
JOB_ID=$(echo $PBS_JOBID | cut -d. -f1)
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="h200"

NODEFILE=$PBS_NODEFILE
NODE_COUNT=$(sort -u $NODEFILE | wc -l)
NUM_NODES=$NODE_COUNT
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
sort -u "$PBS_NODEFILE" | while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done >"$HOSTFILE_NAME"

# training config
SEQ_LENGTH=8192
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=25000

# optimizer config
LR=2.5E-5
MIN_LR=2.5E-6
LR_WARMUP_STEPS=1000
LR_DECAY_STEPS=25000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL=/groups/gag51395/hf_checkpoints/Meta-Llama-3.1-8B/tokenizer.json
CHECKPOINT_DIR=/groups/gag51395/hf_checkpoints/Meta-Llama-3.1-8B
CHECKPOINT_SAVE_DIR="/groups/gag51395/checkpoints/Llama-3.1-8b/llm-recipes/LR_${LR}-minLR_${MIN_LR}-WD_${WEIGHT_DECAY}-GC_${GRAD_CLIP}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH=""

# ja wikipedia
DATA_PATH="${DATA_PATH} 3382423156 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/ja_wiki_merged_text_document"

# job name
JOB_NAME="Llama-3-ABCI-3.0-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -x NCCL_IB_TIMEOUT=22 \
  -x LD_LIBRARY_PATH \
  -x LIBRARY_PATH \
  -x PATH \
  -x INCLUDE \
  -x CUDA_HOME \
  -x CUDA_PATH \
  -x CUDA_NVCC_EXECUTABLE \
  -x CPATH \
  -x CUDNN_PATH \
  -x CUDNN_INCLUDE_DIR \
  -x CUDNN_LIBRARY_DIR \
  -x CUDNN_ROOT_DIR \
  -x NCCL_HOME \
  -x NCCL_INCLUDE_DIR \
  -x NCCL_LIBRARY_DIR \
  -x OMPI_HOME \
  -x MPI_HOME \
  -bind-to none \
  python examples/finetuning.py \
  --seq-length ${SEQ_LENGTH} \
  --sliding-window-size ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --distributed-timeout-minutes 30 \
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
  --wandb-entity "okoge" \
  --wandb-project "ABCI3.0" \
  --wandb-name "${JOB_NAME}"
