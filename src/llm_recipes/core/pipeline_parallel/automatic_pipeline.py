import argparse

import torch
import torch.distributed as torch_distributed
from torch.distributed.pipelining import pipeline, SplitPoint, Schedule1F1B
from torch.distributed.device_mesh import DeviceMesh


# ref: https://github.com/pytorch/PiPPy/blob/main/examples/llama/pippy_llama.py
def automatic_pipeline(
        model: torch.nn.Module,
        micro_batch_inputs: list[dict[str, torch.Tensor]],
        pipeline_parallel_size: int,
        args: argparse.Namespace,
        tp_mesh: DeviceMesh,
    ):
    if torch_distributed.is_initialized():
        rank = torch_distributed.get_rank()
        world_size = torch_distributed.get_world_size()

        num_hidden_size = model.config.num_hidden_layers
        assert num_hidden_size % pipeline_parallel_size == 0
        layer_per_stage = num_hidden_size // pipeline_parallel_size
        split_spec = {
            f"model.layers.{i * layer_per_stage}": SplitPoint.BEGINNING
            for i in range(1, pipeline_parallel_size)
        }

        pipe = pipeline(
            module=model,
            mb_args=(micro_batch_inputs,),
            split_spec=split_spec,
        )

        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        stage = pipe.build_stage(
            stage_index=rank,
            device=device,  # device mesh
            group=tp_mesh.get_group(),
        )

        assert args.global_batch_size % args.micro_batch_size == 0
        assert args.global_batch_size // args.micro_batch_size % args.data_parallel_size == 0
        n_micro_batches = args.global_batch_size // args.micro_batch_size // args.data_parallel_size

        schedule = Schedule1F1B(
            stage=stage,
            n_microbatches=n_micro_batches,
        )

        return schedule

    else:
        raise ValueError("Distributed training is not initialized")
