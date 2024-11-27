import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)


# ref: https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/distributed/tensor_parallelism/fsdp_tp_example.py#L104-L151
def automatic_tensor_split(
    model: torch.nn.Module,
    tp_mesh: DeviceMesh,
):
    """
    This function automatically splits the model into tensor parallelism.
    """
    # Split the model into tensor parallelism
    # ref: embed_tokens: https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/llama/modeling_llama.py#L895
    # ref: norm: https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/llama/modeling_llama.py#L899
    # ref: lm_head: https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/llama/modeling_llama.py#L1111
    model = parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan={
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.norm": SequenceParallel(),
            "model.lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate()
            ),
        }
    )

    for layer_id, transformer_block in enumerate(model.model.layers):
        # MLP: hf <-> pytorch
        # pytorch: https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/distributed/tensor_parallelism/llama2_model.py#L267
        # hf: https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/llama/modeling_llama.py#L310
        layer_tp_plan = {
            "post_attention_layernorm": SequenceParallel(),
            "self_attn": PrepareModuleInput(
                input_layouts=(Shard(1), None),  # type: ignore
                desired_input_layouts=(Replicate(), None),  # type: ignore
            ),
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "input_layernorm": SequenceParallel(),
            "mlp": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "mlp.up_proj": ColwiseParallel(),
        }

        # Adjust attention module to use the local number of heads
        # ref: https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/llama/modeling_llama.py#L683
        # num_heads: https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/llama/modeling_llama.py#L343
        # num_key_value_heads: https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/llama/modeling_llama.py#L345
        attn_layer = transformer_block.self_attn
        attn_layer.num_heads = attn_layer.num_heads // tp_mesh.size()
        attn_layer.num_key_value_heads  = attn_layer.num_key_value_heads  // tp_mesh.size()

        # Custom parallelization plan for the model
        transformer_block = parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan
        )

    return model
