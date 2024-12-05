#!/bin/sh
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=0:01:00:00
#$ -o outputs/convert/$JOB_ID.log
#$ -e outputs/convert/$JOB_ID.log
#$ -p -5

# module load
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

start=5000
end=5000
increment=5000

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/gs/bs/tgh-24IDU/checkpoints/gemma-2-2b/exp2/LR_5.0e-5_MINLR_5.0e-6_WD_0.1_GC_1/${FORMATTED_ITERATION}/model.pt
  OUTPUT_PATH=/gs/bs/tgh-24IDU/checkpoints/fsdp-to-hf/gemma-2-2b/exp2/LR_5.0e-5_MINLR_5.0e-6_WD_0.1_GC_1/${FORMATTED_ITERATION}

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/gs/bs/tga-NII-LLM/hf-checkpoints/gemma-2-2b

  python tools/checkpoint-convert/convert_ckpt.py \
    --hf-base-model-checkpoint-path $BASE_MODEL_CHECKPOINT \
    --hf-tokenizer-path $BASE_MODEL_CHECKPOINT \
    --pytorch-model-checkpoint-path $CHECK_POINT_PATH \
    --out $OUTPUT_PATH \
    --sequence-length 8192
done
