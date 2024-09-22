import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.optim import ZeroRedundancyOptimizer

from llm_recipes.core.tensor_parallel.automatic_tensor import automatic_tensor_split
from llm_recipes.core.pipeline_parallel.automatic_pipeline import automatic_pipeline

from megatron_lm.megatron.global_vars import get_args


# ref: https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/distributed/tensor_parallelism/fsdp_tp_example.py#L83-L93
def init_distributed(
    model: torch.nn.Module,
    rank: int,
    world_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
):
    """
    This is the script for 3D parallelism which combines tensor parallelism, pipeline parallelism, and data parallelism.
    """
    args = get_args()

    # validation
    model_parallel_size = tensor_parallel_size * pipeline_parallel_size
    assert world_size % model_parallel_size == 0, "world_size must be divisible by model_parallel_size"
    assert data_parallel_size == world_size // model_parallel_size, "data_parallel_size must be equal to world_size // model_parallel_size"
    assert tensor_parallel_size <= torch.cuda.device_count(), "tensor_parallel_size must be less than or equal to the number of GPUs in one node"

    # initialize distributed training
    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(pipeline_parallel_size, data_parallel_size, tensor_parallel_size),
        mesh_dim_names=("pp", "dp", "tp"),
    )
    print(f"Device mesh: {device_mesh}", flush=True)

    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]
    pp_mesh = device_mesh["pp"]

    # For TP, input needs to be same across all TP ranks.
    # while for SP, input can be different across all ranks.
    # We will use dp_rank for setting the random seed
    # to mimic the behavior of the dataloader.
    dp_rank = dp_mesh.get_local_rank()

    # tensor parallel
    model = automatic_tensor_split(
        model=model,
        tp_mesh=tp_mesh,
    )

    # data parallel
    # ref: https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/distributed/tensor_parallelism/fsdp_tp_example.py#L154
    model = FSDP(
        module=model,
        process_group=dp_mesh.get_group(),
        use_orig_params=True,
    )

    # pipeline parallel
    pipeline_schedule = automatic_pipeline(
        model=model,
        micro_batch_inputs=...,  # TODO: pass micro_batch
        pipeline_parallel_size=pipeline_parallel_size,
        args=args,
        tp_mesh=tp_mesh,
    )

    return model, pipeline_schedule


# ref: https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html#how-to-use-zeroredundancyoptimizer
def get_distributed_optimizer(
    model: torch.nn.Module,
    lr: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    weight_decay: float,
):
    optimizer = ZeroRedundancyOptimizer(
        params=model.parameters(),
        optimizer_class=torch.optim.adamw.AdamW,
        lr=lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=weight_decay,
    )
    return optimizer
