# LLM Recipes

# Table of Contents

1. [Installation](#installation)
2. [Instruction Tuning](#instruction-tuning)
3. [LLM Continual Pre-Training](#llm-continual-pre-training)

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

## LLM Continual Pre-Training

Docs is coming soon.
