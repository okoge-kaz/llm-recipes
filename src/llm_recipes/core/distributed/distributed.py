import torch
import torch.distributed as torch_distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.optim import ZeroRedundancyOptimizer


# ref: https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/distributed/tensor_parallelism/fsdp_tp_example.py#L83-L93
def init_distributed(
    rank: int,
    world_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
):
    """
    This is the script for 3D parallelism which combines tensor parallelism, pipeline parallelism, and data parallelism.
    """
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
