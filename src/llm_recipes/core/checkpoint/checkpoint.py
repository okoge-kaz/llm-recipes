import time
import torch
import torch.distributed as torch_distributed
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (  # noqa: F401
    FullyShardedDataParallel as FSDP,  # type: ignore
    StateDictType,  # type: ignore
    FullStateDictConfig,  # type:ignore : general model non-sharded, non-flattened params
)
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from pathlib import Path
import os
import gc

from megatron_lm.megatron.global_vars import get_args, get_sampler


def get_local_model_state_dict(model: FSDP) -> dict[str, torch.Tensor]:
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()

    return state_dict


def get_local_optimizer_state_dict(
        model: FSDP, optimizer: torch.optim.Optimizer  # type: ignore
    ) -> dict[str, torch.Tensor]:
    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=None,
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = FSDP.optim_state_dict(model, optimizer)

    return state_dict


def save_dist_model_and_optimizer_state_dict(
    model: FSDP, optimizer: torch.optim.Optimizer, path: str  # type: ignore
) -> None:
    if torch_distributed.get_rank() == 0:
        print(f"Saving model and optimizer state dict to {path}")
    t0 = time.perf_counter()

    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    )
    optim_state_dict = FSDP.optim_state_dict(model, optimizer)
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optim_state_dict,
    }
    dist_cp.save(
        state_dict=state_dict,
        checkpoint_id=path,
    )
    torch_distributed.barrier()
    t1 = time.perf_counter()
    if torch_distributed.get_rank() == 0:
        print(f"Saved model and optimizer state dict to {path}, took {t1 - t0:.2f}s")


def save_model_state_dict(model: FSDP, path: str) -> None:
    state_dict = get_local_model_state_dict(model)
    if torch_distributed.get_rank() == 0:
        print(f"Saving model state dict to {path}")
        torch.save(state_dict, path)
        print(f"Saved model state dict to {path}")
        del state_dict
        gc.collect()


def save_optimizer_state_dict(
        model: FSDP, optimizer: torch.optim.Optimizer, path: str  # type: ignore
    ) -> None:
    state_dict = get_local_optimizer_state_dict(model, optimizer)
    if torch_distributed.get_rank() == 0:
        print(f"Saving optimizer state dict to {path}")
        torch.save(state_dict, path)
        print(f"Saved optimizer state dict to {path}")
        del state_dict
        gc.collect()


def save_scheduler_state_dict(scheduler: torch.optim.lr_scheduler.LRScheduler, path: str) -> None:
    if torch_distributed.get_rank() == 0:
        print(f"Saving scheduler state dict to {path}")
        torch.save(scheduler.state_dict(), path)
        print(f"Saved scheduler state dict to {path}")


def save_sampler_state_dict(sampler: DistributedSampler, path: str) -> None:
    if torch_distributed.get_rank() == 0:
        print(f"Saving sampler indices to {path}")
        torch.save(sampler.state_dict(), path)  # type: ignore
        print(f"Saved sampler indices to {path}")


def save_rng_state(path: str) -> None:
    # PyTorch
    torch_cpu_rng_state = torch.get_rng_state()
    torch_gpu_rng_state = torch.cuda.get_rng_state()
    # Numpy
    import numpy
    np_rng_state = numpy.random.get_state()
    # random
    import random
    py_rng_state = random.getstate()

    # save
    if torch_distributed.get_rank() == 0:
        print(f"Saving RNG states to {path}")
        torch.save({
            'torch_cpu_rng_state': torch_cpu_rng_state,
            'torch_gpu_rng_state': torch_gpu_rng_state,
            'np_rng_state': np_rng_state,
            'py_rng_state': py_rng_state,
        }, path)
        print(f"Saved RNG states to {path}")


def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,  # type: ignore
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: str,
    iteration: int,
) -> None:
    torch_distributed.barrier()
    args = get_args()

    checkpoint_path: str = get_checkpoint_name(path, iteration)
    os.makedirs(checkpoint_path, exist_ok=True)
    if torch_distributed.get_rank() == 0:
        start = time.time()
        print(f"Saving checkpoint to {checkpoint_path}")

    if args.use_dist_ckpt:
        save_dist_model_and_optimizer_state_dict(
            model=model,
            optimizer=optimizer,
            path=checkpoint_path,
        )
    else:
        save_model_state_dict(
            model=model,
            path=f"{checkpoint_path}/model.pt",
        )
        if not args.no_save_optimizer_state:
            save_optimizer_state_dict(
                model=model,
                optimizer=optimizer,
                path=f"{checkpoint_path}/optimizer.pt",
            )

    if args.save_sampler_state:
        sampler = get_sampler()

        save_sampler_state_dict(
            sampler=sampler,
            path=f"{checkpoint_path}/sampler.pt",
        )

    save_scheduler_state_dict(
        scheduler=scheduler,
        path=f"{checkpoint_path}/scheduler.pt",
    )
    save_rng_state(
        path=f"{checkpoint_path}/rng.pt",
    )

    torch_distributed.barrier()

    if torch_distributed.get_rank() == 0:
        with open(f"{path}/latest_iteration.txt", "w") as file:
            file.write(str(iteration))
        print(f"Saved checkpoint to {checkpoint_path}, took {time.time() - start:.2f}s")  # type: ignore


def load_model_state_dict(model: torch.nn.Module, path: str) -> None:
    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        if torch_distributed.get_rank() == 0:
            print(f"No checkpoint found in {path}, skipping model loading")
        return

    latest_checkpoint_path: str = get_checkpoint_name(path, latest_iteration)

    if torch_distributed.get_rank() == 0:
        print(f"Loading model state dict from {latest_checkpoint_path}/model.pt")

    state_dict = torch.load(f"{latest_checkpoint_path}/model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    del state_dict

    if torch_distributed.get_rank() == 0:
        print(f"Loaded model state dict from {latest_checkpoint_path}/model.pt")


def load_dist_model_state_dict(model: FSDP, path: str) -> None:
    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        if torch_distributed.get_rank() == 0:
            print(f"No checkpoint found in {path}, skipping model loading")
        return

    latest_checkpoint_path: str = get_checkpoint_name(path, latest_iteration)

    if torch_distributed.get_rank() == 0:
        print(f"Loading model state dict from {latest_checkpoint_path}")

    t0 = time.perf_counter()
    # ref: https://github.com/pytorch/pytorch/blob/main/test/distributed/checkpoint/test_fsdp_optim_state.py
    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    )
    state_dict = {"model": model.state_dict()}
    dist_cp.load(
        state_dict=state_dict,
        checkpoint_id=latest_checkpoint_path,
    )
    model.load_state_dict(state_dict["model"])

    if torch_distributed.get_rank() == 0:
        print(f"Loaded model state dict from {latest_checkpoint_path}, took {time.perf_counter() - t0:.2f}s")


def load_optimizer_state_dict(
        model: FSDP, optimizer: torch.optim.Optimizer, path: str  # type: ignore
    ) -> None:
    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        if torch_distributed.get_rank() == 0:
            print(f"No checkpoint found in {path}, skipping optimizer loading")
        return

    latest_checkpoint_path: str = get_checkpoint_name(path, latest_iteration)

    if torch_distributed.get_rank() == 0:
        print(f"Loading optimizer state dict from {latest_checkpoint_path}/optimizer.pt")

    state_dict = torch.load(f"{latest_checkpoint_path}/optimizer.pt", map_location="cpu")
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        None,
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = FSDP.optim_state_dict_to_load(model, optimizer, state_dict)
        optimizer.load_state_dict(state_dict)
    del state_dict

    if torch_distributed.get_rank() == 0:
        print(f"Loaded optimizer state dict from {latest_checkpoint_path}/optimizer.pt")


def load_dist_optimizer_state_dict(
        model: FSDP, optimizer: torch.optim.Optimizer, path: str  # type: ignore
    ) -> None:
    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        if torch_distributed.get_rank() == 0:
            print(f"No checkpoint found in {path}, skipping optimizer loading")
        return

    latest_checkpoint_path: str = get_checkpoint_name(path, latest_iteration)

    if torch_distributed.get_rank() == 0:
        print(f"Loading optimizer state dict from {latest_checkpoint_path}")

    t0 = time.perf_counter()
    # ref: https://github.com/pytorch/pytorch/blob/main/test/distributed/checkpoint/test_fsdp_optim_state.py
    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    )
    state_dict = {"model": model.state_dict()}
    dist_cp.load(
        state_dict=state_dict,
        checkpoint_id=latest_checkpoint_path,
    )
    optim_state = load_sharded_optimizer_state_dict(
        model_state_dict=state_dict["model"],
        optimizer_key="optimizer",
        storage_reader=dist_cp.FileSystemReader(latest_checkpoint_path),
        planner=dist_cp.DefaultLoadPlanner()
    )
    flattened_optim_state = FSDP.optim_state_dict_to_load(model, optimizer, optim_state["optimizer"])
    optimizer.load_state_dict(flattened_optim_state)

    if torch_distributed.get_rank() == 0:
        print(f"Loaded optimizer state dict from {latest_checkpoint_path}, took {time.perf_counter() - t0:.2f}s")


def load_scheduler_state_dict(scheduler: torch.optim.lr_scheduler.LRScheduler, path: str) -> None:
    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        return

    latest_checkpoint_path: str = get_checkpoint_name(path, latest_iteration)
    state_dict = torch.load(f"{latest_checkpoint_path}/scheduler.pt", map_location="cpu")
    scheduler.load_state_dict(state_dict)
    del state_dict


def load_sampler_state_dict(sampler: DistributedSampler, path: str) -> None:
    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        return

    latest_checkpoint_path: str = get_checkpoint_name(path, latest_iteration)
    state_dict = torch.load(f"{latest_checkpoint_path}/sampler.pt", map_location="cpu")
    sampler.load_state_dict(state_dict)  # type: ignore
    del state_dict


def load_rng_state_dict(path: str) -> None:
    import numpy
    import random

    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        return

    latest_checkpoint_path: str = get_checkpoint_name(
        path, latest_iteration
    )
    rng_states = torch.load(f"{latest_checkpoint_path}/rng.pt", map_location="cpu")
    torch.set_rng_state(rng_states['torch_cpu_rng_state'])
    torch.cuda.set_rng_state(rng_states['torch_gpu_rng_state'])
    numpy.random.set_state(rng_states['np_rng_state'])
    random.setstate(rng_states['py_rng_state'])

    del rng_states


def read_latest_value(file_path: str) -> int:
    try:
        with open(file_path, "r") as file:
            content = file.read().strip()  # `strip` removes any leading/trailing whitespace
            return int(content)
    except FileNotFoundError:
        if torch_distributed.get_rank() == 0:
            print(f"File not found: {file_path}")
        raise FileNotFoundError
    except ValueError:
        print(f"Unable to convert file content to integer: {file_path}")
        raise ValueError


def get_latest_iteration(path: str) -> int:
    if Path(path).exists():
        try:
            latest_iteration: int = read_latest_value(f"{path}/latest_iteration.txt")
            return latest_iteration
        except (FileNotFoundError, ValueError):
            if torch_distributed.get_rank() == 0:
                print(f"Unable to read latest iteration from {path}/latest_iteration.txt")

    return 0


def get_checkpoint_name(checkpoints_path: str, iteration: int) -> str:
    """Determine the directory name for this rank's checkpoint.

    Args:
        checkpoints_path (str): チェックポイントの保存先
        iteration (int): 学習のiteration

    Returns:
        str: チェエクポイント名
    """
    checkpoint_directory: str = "iter_{:07d}".format(iteration)
    return os.path.join(checkpoints_path, checkpoint_directory)
