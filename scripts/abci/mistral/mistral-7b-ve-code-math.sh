#!/bin/bash
#$ -l rt_AF=4
#$ -l h_rt=3:0:00:00
#$ -j y
#$ -o outputs/mistral-7b-ve/code-math/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# training config
SEQ_LENGTH=4096
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=25000

# optimizer config
LR=2e-5
MIN_LR=2e-6
LR_WARMUP_STEPS=1000
LR_DECAY_STEPS=25000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL=/bb/llm/gaf51275/llama/mistral/swallow-mistral-7B-v0.1-merged-tokenizer-nfkc-16k-hf/tokenizer.model
CHECKPOINT_DIR=/bb/llm/gaf51275/llama/mistral/swallow-mistral-7B-v0.1-merged-tokenizer-nfkc-16k-hf
CHECKPOINT_SAVE_DIR="/bb/llm/gaf51275/llama/checkpoints/mistral-7b-VE/algebraic-stack-the-vault-lr_${LR}-minlr_${MIN_LR}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH=""

# ja okazaki lab cc
DATA_PATH="${DATA_PATH} 9542289837 /bb/llm/gaf51275/llama/datasets/mistral_16k_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_0_text_document"
DATA_PATH="${DATA_PATH} 9417655725 /bb/llm/gaf51275/llama/datasets/mistral_16k_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_1_text_document"
DATA_PATH="${DATA_PATH} 11218202500 /bb/llm/gaf51275/llama/datasets/mistral_16k_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_2_text_document"
DATA_PATH="${DATA_PATH} 14799329550 /bb/llm/gaf51275/llama/datasets/mistral_16k_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_3_text_document"
DATA_PATH="${DATA_PATH} 34519315366 /bb/llm/gaf51275/llama/datasets/mistral_16k_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_4_text_document"

# ja wikipedia
DATA_PATH="${DATA_PATH} 1503207021 /bb/llm/gaf51275/llama/datasets/mistral_16k_Llama2Tokenizer/ja_wiki_merged_text_document"

# en arxiv
DATA_PATH="${DATA_PATH} 4500000000 /bb/llm/gaf51275/llama/datasets/mistral_16k_Llama2Tokenizer/arxiv_text_document"

# en refinedweb
DATA_PATH="${DATA_PATH} 4500000000 /bb/llm/gaf51275/llama/datasets/mistral_16k_Llama2Tokenizer/The_Vault_text_document"

# code and math

# the Vault
DATA_PATH="${DATA_PATH} 5000000000 /bb/llm/gaf51275/llama/datasets/mistral_16k_Llama2Tokenizer/The_Vault_text_document"

# algebraic stack
DATA_PATH="${DATA_PATH} 5000000000 /bb/llm/gaf51275/llama/datasets/mistral_16k_Llama2Tokenizer/algebraic_stack_text_document"

# job name
JOB_NAME="Mistral-7b-VE-code-math-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python examples/finetuning.py \
  --seq-length ${SEQ_LENGTH} \
  --sliding-window-size ${SEQ_LENGTH} \
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
  --wandb-project "abci-mistral-7b" \
  --wandb-name "${JOB_NAME}"
