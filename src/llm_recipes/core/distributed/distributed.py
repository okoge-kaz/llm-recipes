import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.optim import ZeroRedundancyOptimizer

from llm_recipes.core.tensor_parallel.automatic_tensor import automatic_tensor_split
from llm_recipes.core.pipeline_parallel.automatic_pipeline import split_model

from megatron_lm.megatron.global_vars import get_args


# ref: https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/distributed/tensor_parallelism/fsdp_tp_example.py#L83-L93
def init_distributed(
    model: torch.nn.Module,
    rank: int,
    world_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_sharding_size: int,
):
    """
    This is the script for 3D parallelism which combines tensor parallelism, pipeline parallelism, and data parallelism.
    """
    args = get_args()

    # validation
    model_parallel_size = tensor_parallel_size * pipeline_parallel_size
    assert world_size % model_parallel_size == 0, "world_size must be divisible by model_parallel_size"
    assert tensor_parallel_size <= torch.cuda.device_count(), "tensor_parallel_size must be less than or equal to the number of GPUs in one node"
    data_parallel_replicate_size: int = world_size // model_parallel_size // data_parallel_sharding_size
    args.data_parallel_size = data_parallel_replicate_size * data_parallel_sharding_size
    args.data_parallel_replicate_size = data_parallel_replicate_size

    # initialize distributed training
    # ref: https://github.com/pytorch/torchtitan/blob/eef8bb2b1b6f0875ab0581079e1511d51654910e/torchtitan/parallelisms/parallel_dims.py#L55-L56
    if data_parallel_replicate_size > 1 and data_parallel_sharding_size > 1:
        device_mesh: DeviceMesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(
                pipeline_parallel_size,
                data_parallel_replicate_size,
                data_parallel_sharding_size,
                tensor_parallel_size,
            ),
            mesh_dim_names=("pp", "dp_replicate", "dp_shard", "tp"),
        )
    elif data_parallel_replicate_size == 1:
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(
                pipeline_parallel_size,
                data_parallel_sharding_size,
                tensor_parallel_size,
            ),
            mesh_dim_names=("pp", "dp", "tp"),
        )
    elif data_parallel_sharding_size == 1:
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(
                pipeline_parallel_size,
                data_parallel_replicate_size,
                tensor_parallel_size,
            ),
            mesh_dim_names=("pp", "dp", "tp"),
        )
    else:
        raise ValueError(f"Invalid data_parallel_replicate_size and data_parallel_sharding_size: {data_parallel_replicate_size}, {data_parallel_sharding_size}")
    print(f"Device mesh: {device_mesh}", flush=True)

    tp_mesh = device_mesh["tp"]
    if data_parallel_replicate_size > 1 and data_parallel_sharding_size > 1:
        dp_replicate_mesh = device_mesh["dp_replicate"]
        dp_shard_mesh = device_mesh["dp_shard"]
    pp_mesh = device_mesh["pp"]

    # ref: https://github.com/pytorch/torchtitan/blob/eef8bb2b1b6f0875ab0581079e1511d51654910e/torchtitan/parallelisms/parallel_dims.py#L73-L74
    if data_parallel_replicate_size > 1 and data_parallel_sharding_size > 1:
        device_mesh["dp_replicate", "dp_shard"]._flatten(mesh_dim_name="dp")
    # print(f"Device mesh: {device_mesh}", flush=True)
    # print(f"TP mesh: {tp_mesh}, tp_mesh.size()={tp_mesh.size()}", flush=True)

    dp_mesh = device_mesh["dp"]
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()

    vocab_size = model.config.vocab_size
    micro_batch_size = args.micro_batch_size
    seq_length = args.seq_length
    input_ids = torch.randint(  # dummy random input
        0, vocab_size, (micro_batch_size, seq_length)
    )
    labels = torch.randint(  # dummy random labels
        0, vocab_size, (micro_batch_size, seq_length)
    )
    attention_mask = torch.ones(
        micro_batch_size, seq_length
    )

    # pipeline_stage, pipelined_model = split_model(
    #     model=model,
    #     micro_batch_inputs={
    #         "input_ids": input_ids,
    #         "labels": labels,
    #         "attention_mask": attention_mask,
    #     },
    #     pipeline_parallel_size=pipeline_parallel_size,
    #     args=args,
    #     pp_mesh=pp_mesh,
    # )
    # print(f"Pipeline stage: {pipeline_stage}", flush=True)
    # print(f"Pipelined model: {pipelined_model}", flush=True)

    # For TP, input needs to be same across all TP ranks.
    # while for SP, input can be different across all ranks.
    # We will use dp_rank for setting the random seed
    # to mimic the behavior of the dataloader.
    dp_rank = dp_mesh.get_local_rank()

    # tensor parallel
    model = automatic_tensor_split(
        model=model,  # pipelined_model, pipeline_stageのどちらを渡すべき?
        tp_mesh=tp_mesh,
    )
    # print(f"Tensor parallel model: {model}", flush=True)

    # data parallel
    # ref: https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/distributed/tensor_parallelism/fsdp_tp_example.py#L154
    shard_model = FSDP(
        module=model,
        device_mesh=dp_mesh,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )
    # print(f"Shard model: {shard_model}", flush=True)

    return shard_model, device_mesh


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
