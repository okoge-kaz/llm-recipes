# LLM Recipes

# Table of Contents

1. [Installation](#installation)
2. [Instruction Tuning](#instruction-tuning)
3. [LLM Continual Pre-Training](#llm-continual-pre-training)
4. [Support Models](#support-models)

## Installation

To install the package, run the following command:

```bash
pip install -r requirements.txt
```

If you want to use the library in multi-nodes, you need to install the below packages:

```bash
module load openmpi/4.x.x

pip install mpi4py
```

### FlashAttention

To install the FlashAttention, run the following command: (GPU is required)

```bash
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation
```

### ABCI

If you use [ABCI](https://abci.ai/) to run the experiments, install scripts are available in `llm-recipes/install.sh`.

## Instruction Tuning

[scripts/abci/instruction](scripts/abci/instruction) contains the scripts to run instruction tunings on ABCI.

If you want to use custom instructions, you need to modify the `src/llama_recipes/datasets/alpaca_dataset.py`.

## LLM Continual Pre-Training

[scripts/abci/](scripts/abci/) contains the scripts to run LLM continual pre-training on ABCI.
7B, 13B, 70B directories contain the scripts to run the experiments with the corresponding model size (Llama 2).

## Support Models

- [meta Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [mistral 7b](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [swallow](https://huggingface.co/tokyotech-llm/Swallow-70b-hf)
