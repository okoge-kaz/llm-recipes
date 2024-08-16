<div align="center">

llm-recipes
===========================
<h4>User-friendly tool for seamless continual pre-training of Large Language Models</h4>

<img src="images/llm-recipes-logo.webp" alt="llm-recipes" width="300px">

<div align="left">

llm-recipes is a tool designed to make the continual pre-training of Large Language Models (LLMs) easy and efficient. With an intuitive interface and flexible configuration options, researchers and developers can effortlessly manage training on any model or dataset. The tool supports distributed training on large GPU clusters and offers extensive customization, enabling users to leverage cutting-edge techniques with ease.

What sets llm-recipes apart is its seamless integration with Hugging Face Transformers, allowing you to continue pre-training or perform instruction tuning on Dense LLMs (non-MoE models) with minimal changes. This means there’s no need to convert checkpoints or deal with complex workflows—just focus on refining your model.

| Feature                         | llm-recipes | llama-recipes | torchtune |
|---------------------------------|-------------|---------------|-----------|
| **SFT(Supervised Fine-Tuning)** | ✅          | ✅            | ✅        |
| **Continual Pre-Training**      | ✅          | ✅            | ✅        |
| **DPO(Direct Preference Optimization)** | ✅          | ❌            | ❌        |
| **Llama Models Support**        | ✅          | ✅            | ✅       |
| **Non-Llama Models Support**    | ✅          | ❌            | ❌       |
| **Multi-Node Support**          | ✅          | ✅            | ❌       |

# Table of Contents

- [Installation](#installation)
  - [Multi-node Support](#multi-node-support)
  - [FlashAttention](#flashattention)
- [Usage](#usage)
  - [LLM Instruction Tuning](#llm-instruction-tuning)
  - [LLM Continual Pre-Training](#llm-continual-pre-training)
  - [LLM DPO](#llm-dpo)
- [Checkpoint formats](#checkpoint-formats)
  - [llm-recipes format](#llm-recipes-format)
  - [PyTorch format to Hugging Face format](#pytorch-format-to-hugging-face-format)
  - [PyTorch distributed format to Hugging Face format](#pytorch-distributed-format-to-hugging-face-format)
- [Inference](#inference)
- [Training Speed and Scalability](#training-speed-and-scalability)
- [Projects Using llm-recipes](#projects-using-llm-recipes)
- [Citation](#citation)

## Installation

This package has been tested with Python 3.10 and 3.11. The recommended environment is with CUDA Toolkit 12.1.

To install the required packages, simply run:

```bash
pip install -r requirements.txt
```
> Note: The requirements.txt assumes that CUDA Toolkit 12.1 is installed on your system.

### Multi-node Support

For multi-node support, ensure you have the following dependencies installed:

```bash
module load openmpi/4.x.x

pip install mpi4py
```

### FlashAttention

For GPU-accelerated FlashAttention, follow these steps:

```bash
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation
```

## Usage

### LLM Instruction Tuning

#### 1. **Data Preparation**

Prepare your data in the below format and save it as a JSONL file:

```jsonl
{
  "input": [
    {
      "role": "user",
      "content": "What is the weather like today?"
    }
  ],
  "output": {
    "role": "assistant",
    "content": "The weather is sunny with a high of 25 degrees."
  }
}
```

#### 2. **Change Dataset Class**

Please modify the `Dataset` class in `src/llama_recipes/utils/instruction_tuning.py` to adjust to the model's expected format.
But, almost all the models have chat templates, so you may not need to change the `Dataset` class.

#### 3. **Indexing**

To load dataset efficiently, create an index file using the following command:

```bash
python tools/pre-process/index_dataset.py \
  --data-file-path <path-to-jsonl-file>
```

After indexing, `.index_cache` directory will be created in the same directory as the JSONL file.

#### 4. **Training**

We provide an example script for instruction tuning for Llama-3-8B in `scripts/tsubame/instruct/Llama-3-8B/Llama-3-8B-instruct-v0.2.sh`.
You can modify the script to suit your needs.

### LLM Continual Pre-Training

#### 1. **Data Preparation**

Prepare your data in the below format and save it as a JSONL file:

```jsonl
{
  "text": "What is the weather like today?\nThe weather is sunny with a high of 25 degrees."
}
```

#### 2. **Tokenize Data**

Tokenize your data using the tokenizer provided by the model you are using.
For example, to tokenize data for Codestral(Mistral-AI), run the following command:

```bash
DATASET_DIR=/path/to/datasets/samples
OUTPUT_DIR=/path/to/datasets/debug/Codestral-22B-v0.1

mkdir -p ${OUTPUT_DIR}

python megatron_lm/tools/preprocess_data.py \
  --input ${DATASET_DIR}/ja_wiki.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /path/to/hf_checkpoints/Codestral-22B-v0.1/tokenizer.model \
  --append-eod \
  --workers 64
```

#### 3. **Training**

We support Llama-2, Llama-3, Llama-3.1, Mistral, Codestral, Phi-3, Yi-1.5, and gemma-2.
If you want to continually pre-train or instruction tune other models, you should modify `src/llama_recipes/get_models.py` and `src/llama_recipes/get_model_decoder_layer.py`.

We provide example scripts for continual pre-training for codestral-22B in `scripts/gcp/codestral-22b.sh`.
You can modify the script to suit your needs.

### LLM DPO

we experimentally support DPO, but it is not fully tested.
The documentation will be updated soon.


## Checkpoint formats

### llm-recipes format

llm-recipes supports 2 types of checkpoints: PyTorch format and PyTorch distributed format.
The PyTorch format is a simple checkpoint format. The example of the PyTorch format is as follows:

```bash
model.pt  optimizer.pt  rng.pt  sampler.pt  scheduler.pt
```
PyTorch distributed format is a checkpoint format that can be distributed-loaded using `torch.distributed`.
The example of the PyTorch distributed format is as follows:

```bash
__0_0.distcp  __1_0.distcp  __2_0.distcp  __3_0.distcp  __4_0.distcp  __5_0.distcp  __6_0.distcp  __7_0.distcp  rng.pt  sampler.pt  scheduler.pt
```

### PyTorch format to Hugging Face format

You can convert the PyTorch format to the Hugging Face format using the following command:

```bash
ITERATION=1000
FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

CHECK_POINT_PATH=/path/to/train/checkpoint/${FORMATTED_ITERATION}/model.pt
OUTPUT_PATH=/path/to/converted/checkpoint/${FORMATTED_ITERATION}

mkdir -p $OUTPUT_PATH

BASE_MODEL_CHECKPOINT=/path/to/huggingface-checkpoint/Llama-2-7b-hf

python tools/checkpoint-convert/convert_ckpt.py \
  --model $BASE_MODEL_CHECKPOINT \
  --ckpt $CHECK_POINT_PATH \
  --out $OUTPUT_PATH \
  --sequence-length 4096
```

### PyTorch distributed format to Hugging Face format

You can convert the PyTorch distributed format to the Hugging Face format using the following command:

```bash
  ITERATION=1000
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/path/to/fsdp/checkpoint/${FORMATTED_ITERATION}
  OUTPUT_PATH=/path/to/converted-hf-checkpoint/${FORMATTED_ITERATION}

  echo "convert FSDP ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/path/to/hf-checkpoints/Meta-Llama-3-8B-Instruct

  python tools/checkpoint-convert/convert_fsdp.py \
  --hf-base-model-path $BASE_MODEL_CHECKPOINT \
  --tokenizer-path $BASE_MODEL_CHECKPOINT \
  --fsdp-checkpoint-path $CHECK_POINT_PATH \
  --checkpoint-output-path $OUTPUT_PATH \
  --sequence-length 8192
```

## Inference

After checkpoint conversion, you can use the Hugging Face Transformers library to load the converted checkpoint and perform inference.

The following is an example of how to do inference using the converted checkpoint (huggingface format):

```bash
python tools/inference/inference.py \
  --model-path /path/to/converted/iter_0004000 \
  --tokenizer-path /path/to/tokenizer/path \
  --prompt "Tokyo is the capital of"
```

## Training Speed and Scalability

We are currently working on improving the training speed and scalability of llm-recipes.
We will update this section with more information soon.

## Projects Using llm-recipes

Below are some of the projects where we have directly used llm-recipes:

- [Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities](https://arxiv.org/abs/2404.17790)
- [Building a Large Japanese Web Corpus for Large Language Models](https://arxiv.org/abs/2404.17733)
- [Turing(company)](https://tur.ing/en)'s [GENIAC](https://www.meti.go.jp/english/policy/mono_info_service/geniac/index.html) project (SFT training)

## Citation

we are current submitting the paper to SC24 workshop, and the citation will be updated soon.

```bibtex
@software{Fujii_llm-recipes_2024,
author = {Kazuki Fujii and Taishi Nakamura and Rio Yokota},
month = may,
title = {{llm-recipes}},
url = {https://github.com/okoge-kaz/llm-recipes},
version = {1.0.0},
year = {2024}
}
```
