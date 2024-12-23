from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch
from typing import Any
import math


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)  # seed is defined in the parent class
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch: int = epoch

    def state_dict(self) -> dict[str, Any]:
        return {"current_epoch": self.current_epoch, "generator_state": self.generator.get_state()}

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict["current_epoch"]
        self.generator.set_state(state_dict["generator_state"])

    def __iter__(self):
        # Set the seed for shuffling based on the current epoch and generator state
        g = torch.Generator()
        g.set_state(self.generator.get_state())
        g.manual_seed(self.current_epoch)

        # dataset
        dataset_length: int = len(self.dataset)  # type: ignore

        # The rest of the implementation is similar to the original DistributedSampler
        # Get the number of samples per process and the start index for the current process
        num_samples = int(math.ceil(dataset_length * 1.0 / self.num_replicas))
        total_size = num_samples * self.num_replicas
        self.num_samples = num_samples

        # Shuffle dataset or generate a linear sequence
        if self.shuffle:
            indices = torch.randperm(dataset_length, generator=g).tolist()
        else:
            indices = list(range(dataset_length))

        # Add extra samples to make it evenly divisible
        indices += indices[: (total_size - len(indices))]
        assert len(indices) == total_size

        # Subsample for the current process
        offset = self.rank * self.num_samples
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        # Update the generator state after shuffling
        self.generator.set_state(g.get_state())

        return iter(indices)
