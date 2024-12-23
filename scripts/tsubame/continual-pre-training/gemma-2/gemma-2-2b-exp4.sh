#!/bin/sh
#$ -cwd
#$ -l node_f=8
#$ -l h_rt=1:00:00:00
#$ -o outputs/gemma-2-2b/$JOB_ID.log
#$ -e outputs/gemma-2-2b/$JOB_ID.log
#$ -p -5

# module load
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

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
GLOBAL_BATCH_SIZE=256
TRAIN_STEPS=25000

# optimizer config
LR=1.0e-4
MIN_LR=1.0e-5
LR_WARMUP_STEPS=1000
LR_DECAY_STEPS=$TRAIN_STEPS
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint
TOKENIZER_PATH=/gs/bs/tga-NII-LLM/hf-checkpoints/gemma-2-2b/tokenizer.model
CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/hf-checkpoints/gemma-2-2b
CHECKPOINT_SAVE_DIR="/gs/bs/tgh-24IDU/checkpoints/gemma-2-2b/exp4/LR_${LR}_MINLR_${MIN_LR}_WD_${WEIGHT_DECAY}_GC_${GRAD_CLIP}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# dataset
TRAIN_DATA_PATH=""

# japanese wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2956562564 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/wiki/ja_wiki_merged_text_document"

# japanese llm top10
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15325170717 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/filter-v2-gemma-top10/dump_0_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11171813236 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/filter-v2-gemma-top10/dump_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 22573934122 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/filter-v2-gemma-top10/dump_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 20742716950 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/filter-v2-gemma-top10/dump_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7157425875 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/filter-v2-gemma-top10/dump_4_text_document"

# japanese wiki like
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 12457136191 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/merge_filter-v2-wiki-top10/merged_2013_2017_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7328307646 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/merge_filter-v2-wiki-top10/merged_2018_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9208865146 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/merge_filter-v2-wiki-top10/merged_2019_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8423862478 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/merge_filter-v2-wiki-top10/merged_2020_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9096928420 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/merge_filter-v2-wiki-top10/merged_2021_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7073479749 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/merge_filter-v2-wiki-top10/merged_2022_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4483796906 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/merge_filter-v2-wiki-top10/merged_2023_text_document"

# je-en parallel corpus
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 769992751 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/Laboro-ParaCorpus/default_plain_text_format_text_document"

# en wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4560810471 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/wiki/en_wiki_merged_train_text_document"

# en cosmopedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2552082704 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/cosmopedia/cosmopedia_automathtext_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 40688636 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/cosmopedia/cosmopedia_khanacademy_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 189812324 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/cosmopedia/cosmopedia_openstax_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1829599180 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/cosmopedia/cosmopedia_stanford_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5243001898 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/cosmopedia/cosmopedia_stories_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2653895178 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/cosmopedia/cosmopedia_web_samples_v1_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2173280620 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/cosmopedia/cosmopedia_web_samples_v2_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 317639460 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/cosmopedia/cosmopedia_wikihow_train_text_document"

# en dclm
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 969356476.7 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/dclm-baseline-1.0/merged_one_tenth/global-shard_01_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 970120780.9 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/dclm-baseline-1.0/merged_one_tenth/global-shard_02_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 937442507.8 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/dclm-baseline-1.0/merged_one_tenth/global-shard_03_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 970902315.2 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/dclm-baseline-1.0/merged_one_tenth/global-shard_04_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 971438901.3 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/dclm-baseline-1.0/merged_one_tenth/global-shard_05_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 970010729.8 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/dclm-baseline-1.0/merged_one_tenth/global-shard_06_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 971201773.3 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/dclm-baseline-1.0/merged_one_tenth/global-shard_07_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 970774107.7 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/dclm-baseline-1.0/merged_one_tenth/global-shard_08_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 969184838.5 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/dclm-baseline-1.0/merged_one_tenth/global-shard_09_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 968764346.8 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/dclm-baseline-1.0/merged_one_tenth/global-shard_10_of_10_text_document"

# code stack v2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 12937299736 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/bigcode/the-stack-v2-train-smol-ids/random_sample0.1_merge/the-stack-v2-train-smol-ids-00_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19062700264 /gs/fs/jh160041/datasets/Swallow/binarized/gemma-2_original_transformers-4.45.2/bigcode/the-stack-v2-train-smol-ids/random_sample0.1_merge/the-stack-v2-train-smol-ids-01_text_document"

# job name
JOB_NAME="Gemma-2-2b-Exp-4-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none \
  -x NCCL_IB_TIMEOUT=22 \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -x LD_LIBRARY_PATH \
  -x PATH \
  python train_llm.py \
  --seq-length ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_PATH} \
  --data-path ${TRAIN_DATA_PATH} \
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
  --save-interval 1000 \
  --eval-interval 1000 \
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
  --continual-pretraining \
  --use-mpi \
  --wandb-entity "prj-jalm" \
  --wandb-project "gemma-2-2b" \
  --wandb-name "${JOB_NAME}"
