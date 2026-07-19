"""Tests for src/datasets/adapter.py's TranslationDataset."""

import numpy as np
import torch

from src.datasets.adapter import TranslationDataset


class _FakeBaseDataset:
    """Stands in for a real loader (sarptical.py etc.) -- same {"sar", "optical"} (H,W,C) numpy
    dict contract described in src/datasets/common.py, without needing a real download."""

    def __init__(self, n=4, sar_shape=(16, 16, 2), optical_shape=(16, 16, 3)):
        rng = np.random.default_rng(0)
        self._samples = [
            {
                "sar": rng.normal(loc=100, scale=20, size=sar_shape).astype(np.float32),
                "optical": rng.uniform(0, 255, size=optical_shape).astype(np.float32),
            }
            for _ in range(n)
        ]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]


class TestTranslationDataset:
    def test_len_matches_base_dataset(self):
        dataset = TranslationDataset(_FakeBaseDataset(n=7))
        assert len(dataset) == 7

    def test_returns_channel_first_tensors(self):
        dataset = TranslationDataset(_FakeBaseDataset(sar_shape=(16, 16, 2), optical_shape=(16, 16, 3)))
        sample = dataset[0]
        assert sample["sar"].shape == (2, 16, 16)
        assert sample["optical"].shape == (3, 16, 16)
        assert isinstance(sample["sar"], torch.Tensor)
        assert isinstance(sample["optical"], torch.Tensor)

    def test_output_is_tanh_range(self):
        dataset = TranslationDataset(_FakeBaseDataset())
        sample = dataset[0]
        assert sample["sar"].min() >= -1.0
        assert sample["sar"].max() <= 1.0
        assert sample["optical"].min() >= -1.0
        assert sample["optical"].max() <= 1.0

    def test_works_with_a_real_dataloader_and_batches_correctly(self):
        dataset = TranslationDataset(_FakeBaseDataset(n=4, sar_shape=(16, 16, 2), optical_shape=(16, 16, 3)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        assert batch["sar"].shape == (2, 2, 16, 16)
        assert batch["optical"].shape == (2, 3, 16, 16)
