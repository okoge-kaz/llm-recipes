import argparse
from typing import Optional, Callable

import torch
import torch.distributed as torch_distributed
from torch.distributed.pipelining import pipeline, SplitPoint, Schedule1F1B
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import (
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)
from torch.distributed.pipelining.stage import _PipelineStageBase

from megatron_lm.megatron.global_vars import get_args


def build_pipeline_schedule(
    pipeline_stage,
    n_micro_batches: Optional[int] = None,
    loss_func: Optional[Callable] = None,
):
    args = get_args()

    looped_schedule = args.pipeline_parallel_schedule in ["interleaved_1f1b"]

    if args.pipeline_parallel_schedule == "1f1b":
        schedule_class = Schedule1F1B
    elif args.pipeline_parallel_schedule == "gpipe":
        schedule_class = ScheduleGPipe
    elif args.pipeline_parallel_schedule == "interleaved_1f1b":
        schedule_class = ScheduleInterleaved1F1B
    else:
        raise ValueError(f"Invalid pipeline_parallel_schedule: {args.pipeline_parallel_schedule}")

    if n_micro_batches is None:
        n_micro_batches = args.global_batch_size // args.micro_batch_size // args.data_parallel_size

    schedule = schedule_class(
        pipeline_stage if looped_schedule else pipeline_stage[0],
        n_microbatches=n_micro_batches,  # type: ignore
        loss_fn=loss_func,
    )

    return schedule



# ref: https://github.com/pytorch/PiPPy/blob/main/examples/llama/pippy_llama.py
def split_model(
        model: torch.nn.Module,
        micro_batch_inputs: dict[str, torch.Tensor],
        pipeline_parallel_size: int,
        args: argparse.Namespace,
        pp_mesh: DeviceMesh,
    ):
    args = get_args()

    if torch_distributed.is_initialized():
        rank = torch_distributed.get_rank()
        world_size = torch_distributed.get_world_size()
        pp_rank = pp_mesh.get_local_rank()

        num_hidden_size = model.config.num_hidden_layers
        assert num_hidden_size % pipeline_parallel_size == 0
        layer_per_stage = num_hidden_size // pipeline_parallel_size
        split_spec = {
            f"model.layers.{i * layer_per_stage}": SplitPoint.BEGINNING
            for i in range(1, pipeline_parallel_size)
        }

        pipe = pipeline(
            module=model,
            mb_args=(micro_batch_inputs["input_ids"],),
            split_spec=split_spec,
        )
        print(f"Pipeline: {pipe}", flush=True)

        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        stage = pipe.build_stage(
            stage_index=pp_rank,
            device=device,
            group=pp_mesh.get_group(),
        )

        assert args.global_batch_size % args.micro_batch_size == 0
        assert args.global_batch_size // args.micro_batch_size % args.data_parallel_size == 0

        return stage, pipe

    else:
        raise ValueError("Distributed training is not initialized")
