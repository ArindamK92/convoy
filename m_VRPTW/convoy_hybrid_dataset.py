"""Dataset utilities for hybrid RL4CO runner."""

from __future__ import annotations

import torch

from tensordict import TensorDict
from torch.utils.data import Dataset


class FixedInstanceDataset(Dataset):
    """Dataset that repeats one fixed TensorDict instance."""

    def __init__(self, sample_td: TensorDict, size: int):
        self.sample_td = sample_td.clone()
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx):
        del idx
        return self.sample_td.clone()

    def collate_fn(self, batch):
        return torch.stack(batch, dim=0)
