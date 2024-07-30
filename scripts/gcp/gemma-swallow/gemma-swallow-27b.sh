#!/bin/bash
#SBATCH --job-name=gemma
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --nodes 8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=3-12:00:00
#SBATCH --output=outputs/gemma/%x-%j.out
#SBATCH --error=outputs/gemma/%x-%j.out

set -e

# module load
module load turing/cuda/12.1
module load turing/cudnn/8.9.7
module load turing/nccl/2.20.5
module load turing/hpcx/2.17.1

# open file limit
ulimit -n 65536 1048576

# python virtualenv
source .env/bin/activate

# Important TCPX environment variables
UDS_PATH="/run/tcpx-${SLURM_JOB_ID}"

# Only use TCPX for multi-node jobs.
[[ "${SLURM_JOB_NUM_NODES}" -gt 1 ]] && export USE_TCPX=yes || export USE_TCPX=no

# Only use TCPX for multi-node jobs.
if [[ ${USE_TCPX} = "yes" ]]; then
  # Set up NCCL Environment variables
  export NCCL_NET=GPUDirectTCPX_v7
  # These network interfaces use Ubuntu's consistent naming scheme. See
  # https://manpages.ubuntu.com/manpages/focal/man7/systemd.net-naming-scheme.7.html
  export NCCL_SOCKET_IFNAME=enp0s12
  export NCCL_GPUDIRECTTCPX_CTRL_DEV=enp0s12
  export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=enp6s0,enp12s0,enp134s0,enp140s0
  export NCCL_CROSS_NIC=0
  export NCCL_ALGO=Ring
  export NCCL_PROTO=Simple
  export NCCL_NSOCKS_PERTHREAD=4
  export NCCL_SOCKET_NTHREADS=1
  export NCCL_DYNAMIC_CHUNK_SIZE=524288
  export NCCL_P2P_NET_CHUNKSIZE=524288
  export NCCL_P2P_PCI_CHUNKSIZE=524288
  export NCCL_P2P_NVL_CHUNKSIZE=1048576
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export NCCL_NET_GDR_LEVEL=PIX
  export NCCL_P2P_PXN_LEVEL=0
  export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=${UDS_PATH}
  export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=500000
  export NCCL_GPUDIRECTTCPX_TX_BINDINGS="enp6s0:8-21,112-125;enp12s0:8-21,112-125;enp134s0:60-73,164-177;enp140s0:60-73,164-177"
  export NCCL_GPUDIRECTTCPX_RX_BINDINGS="enp6s0:22-35,126-139;enp12s0:22-35,126-139;enp134s0:74-87,178-191;enp140s0:74-87,178-191"

  export LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH}
else
  unset NCCL_NET
fi

# distributed settings
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="H100"

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# training config
SEQ_LENGTH=8192
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=512
TRAIN_STEPS=25000

# optimizer config
LR=1.5E-5
MIN_LR=1.5E-6
LR_WARMUP_STEPS=1000
LR_DECAY_STEPS=25000
WEIGHT_DECAY=0.1
GRAD_CLIP=1
# model config
TOKENIZER_MODEL=/home/ext_kazuki_fujii_turing_motors_c/hf-checkpoints/gemma-2-27b/tokenizer.model
CHECKPOINT_DIR=/home/ext_kazuki_fujii_turing_motors_c/hf-checkpoints/gemma-2-27b
CHECKPOINT_SAVE_DIR="/data/checkpoints/gemma-2-27b/LR${LR}-MINLR${MIN_LR}-WARMUP${LR_WARMUP_STEPS}-WD${WEIGHT_DECAY}-GC${GRAD_CLIP}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/data/gemma_datasets/gemma-2_original_transformers-4.42.4

TRAIN_DATA_PATH=""

# ja swallow corpus
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9651717149 ${DATASET_DIR}/split_0_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9509783737 ${DATASET_DIR}/split_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11318518471 ${DATASET_DIR}/split_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14904913186 ${DATASET_DIR}/split_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 34418569125 ${DATASET_DIR}/split_4_text_document"

# ja wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1478281282 ${DATASET_DIR}/ja_wiki_merged_text_document"

# ja-en laboro
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 769992751 ${DATASET_DIR}/default_plain_text_format_text_document"

# en wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1890321820 ${DATASET_DIR}/en_wiki_merged_train_text_document"

# en refinedweb
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1890321820 ${DATASET_DIR}/lumi_en_falcon_merge_text_document"

# en cosmopedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1276041352 ${DATASET_DIR}/cosmopedia_automathtext_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 20344318 ${DATASET_DIR}/cosmopedia_khanacademy_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 94906162 ${DATASET_DIR}/cosmopedia_openstax_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 914799590 ${DATASET_DIR}/cosmopedia_stanford_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2621500949 ${DATASET_DIR}/cosmopedia_stories_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 158819730 ${DATASET_DIR}/cosmopedia_wikihow_train_text_document"

# code
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4540584279 ${DATASET_DIR}/algebraic-stack_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4540584279 ${DATASET_DIR}/proof-pile-2-train_merged_open-web-math_text_document"


# job name
JOB_NAME="gemma-2-turing-swallow-gcp-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none \
  -x LD_LIBRARY_PATH \
  -x PATH \
  python examples/finetuning.py \
  --seq-length ${SEQ_LENGTH} \
  --sliding-window-size ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --data-path ${TRAIN_DATA_PATH} \
  --split 989,10,1 \
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
  --save-interval 250 \
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
  --wandb-entity "turing-geniac" \
  --wandb-project "gemma-2-turing-swallow" \
  --wandb-name "${JOB_NAME}"
