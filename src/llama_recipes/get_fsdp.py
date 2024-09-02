from torch.distributed.fsdp import ShardingStrategy  # type: ignore
from megatron_lm.megatron.global_vars import get_args


def get_sharding_strategy() -> ShardingStrategy:
    args = get_args()

    if args.sharding_strategy == "FULL_SHARD":
        return ShardingStrategy.FULL_SHARD
    elif args.sharding_strategy == "SHARD_GRAD_OP":
        return ShardingStrategy.SHARD_GRAD_OP
    elif args.sharding_strategy == "NO_SHARD":
        return ShardingStrategy.NO_SHARD
    elif args.sharding_strategy == "HYBRID_SHARD":
        # TODO: https://pytorch.org/tutorials/recipes/distributed_device_mesh.html#how-to-use-devicemesh-with-hsdp
        # support device mesh
        # ref: https://github.com/meta-llama/llama-recipes/blob/778e31e35cfbe385a31b3a94b794e3f75e276d1a/src/llama_recipes/finetuning.py#L160
        return ShardingStrategy.HYBRID_SHARD
    elif args.sharding_strategy == "_HYBRID_SHARD_ZERO2":
        return ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise NotImplementedError
