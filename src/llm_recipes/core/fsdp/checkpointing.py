from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from llm_recipes.training.get_model_decoder_layer import get_model_decoder_layer

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)
from llm_recipes.core.fsdp.distributed import print_rank_0


def apply_fsdp_checkpointing(model, model_name: str) -> None:
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print_rank_0("--> applying fsdp activation checkpointing...")

    check_fn = lambda submodule: isinstance(  # noqa: E731
        submodule, get_model_decoder_layer(
            model_name=model_name
        )
    )

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
