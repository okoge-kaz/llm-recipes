#!/bin/bash
#$ -l rt_AF=4
#$ -l h_rt=10:00:00:00
#$ -j y
#$ -o outputs/phi-3/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module use /groups/gag51395/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.17/2.17.1-1
module load hpcx/2.12
module load gcc/11.4.0

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

MICRO_BATCH_SIZE=4
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
TOKENIZER_MODEL=/bb/llm/gaf51275/hf-checkpoints/Phi-3-medium-4k-instruct/tokenizer.model
CHECKPOINT_DIR=/bb/llm/gaf51275/hf-checkpoints/Phi-3-medium-4k-instruct
CHECKPOINT_SAVE_DIR="/bb/llm/gaf51275/checkpoints/Phi-3-medium-4k/LR_${LR}-minLR_${MIN_LR}-WD_${WEIGHT_DECAY}-GC_${GRAD_CLIP}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH=""

# Swallow v1
DATA_PATH="${DATA_PATH} 9108171060 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/split_0_text_document"
DATA_PATH="${DATA_PATH} 9017389663 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/split_1_text_document"
DATA_PATH="${DATA_PATH} 10781891782 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/split_2_text_document"
DATA_PATH="${DATA_PATH} 14229527811 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/split_3_text_document"
DATA_PATH="${DATA_PATH} 33251122086 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/split_4_text_document"

# ja wikipedia
DATA_PATH="${DATA_PATH} 2659052072 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/ja_wiki_merged_text_document"

# parallel corpus
DATA_PATH="${DATA_PATH} 1265915426 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/default_plain_text_format_text_document"

# en wikipedia
DATA_PATH="${DATA_PATH} 1400935123 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/en_wiki_merged_train_text_document"

# en refinedweb
DATA_PATH="${DATA_PATH} 1400935123 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/lumi_en_falcon_merge_text_document"

# en cosmopedia
DATA_PATH="${DATA_PATH} 1394911660 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/cosmopedia_automathtext_train_text_document"
DATA_PATH="${DATA_PATH} 22852028 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/cosmopedia_khanacademy_train_text_document"
DATA_PATH="${DATA_PATH} 115215400 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/cosmopedia_openstax_train_text_document"
DATA_PATH="${DATA_PATH} 1120661316 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/cosmopedia_stanford_train_text_document"
DATA_PATH="${DATA_PATH} 3131907229 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/cosmopedia_stories_train_text_document"
DATA_PATH="${DATA_PATH} 195599284 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/cosmopedia_wikihow_train_text_document"

# code algebraic stack
DATA_PATH="${DATA_PATH} 10903912936 /bb/llm/gaf51275/datasets/Phi-3_original_transformers-4.40.1/algebraic-stack_text_document"


# job name
JOB_NAME="Phi-3-ABCI-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
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
  --adam-eps 1e-8 \
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
  --wandb-project "Phi-3-medium" \
  --wandb-name "${JOB_NAME}"
