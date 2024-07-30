import torch

from torch.distributed.fsdp.api import (
    MixedPrecision,
)

# requires grad scaler in main loop
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float32,  # Use float32 for gradient communication. (Llama-3, Megatron_LM like)
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.float32,  # Use float32 for gradient communication. (Llama-3, Megatron_LM like)
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)

bfSixteen_mixed = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,  # Use float32 for gradient communication. (Llama-3, Megatron_LM like)
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)
