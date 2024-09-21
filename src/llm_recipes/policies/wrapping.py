import functools

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from llm_recipes.training.get_model_decoder_layer import get_model_decoder_layer


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=int(min_params)
    )
    return num_wrap_policy


def get_decoder_layer_wrapper(model_name: str):
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            get_model_decoder_layer(
                model_name=model_name
            )
        },
    )

    return llama_auto_wrap_policy
