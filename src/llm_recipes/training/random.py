import torch
import numpy as np
import random
from torch.distributed.device_mesh import DeviceMesh
from typing import Optional


def set_seed(seed: int, dp_mesh: Optional[DeviceMesh]=None) -> None:
    # Set the seeds for reproducibility
    # ref: https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/distributed/tensor_parallelism/fsdp_tp_example.py#L172
    if dp_mesh is not None:
        rank = dp_mesh.get_local_rank()
        seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
