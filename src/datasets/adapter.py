"""
Adapts any of this package's dataset loaders (bigearthnet.py, sen12ms.py, sen1_2.py,
sarptical.py — all of which return the shared {"sar": (H,W,C), "optical": (H,W,C)} numpy dict
described in src/datasets/common.py) into a torch.utils.data.Dataset ready to hand a DataLoader
straight to src/models/ generators: normalized to tanh's [-1, 1] range, channel-first tensors.

Kept as one thin adapter rather than duplicating this normalization+conversion step inside each
dataset loader, so it's the training script (scripts/train_baseline.py), not the loaders
themselves, that decides a sample needs to become model-ready -- the loaders' own job stays
"faithfully read what's on disk" (see e.g. sarptical.py's docstring on why its two real-format
discrepancies were resolved by checking the actual files, not assumed).
"""

from __future__ import annotations

import torch.utils.data

from src.datasets.common import hwc_to_chw_tensor, normalize_to_tanh_range


class TranslationDataset(torch.utils.data.Dataset):
    """Wraps a base dataset (anything with __len__ and __getitem__ returning a {"sar", "optical"}
    dict of (H, W, C) numpy arrays) into normalized (C, H, W) tanh-range tensors."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        return {
            "sar": hwc_to_chw_tensor(normalize_to_tanh_range(sample["sar"])),
            "optical": hwc_to_chw_tensor(normalize_to_tanh_range(sample["optical"])),
        }
