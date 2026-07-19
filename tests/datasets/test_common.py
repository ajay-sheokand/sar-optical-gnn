"""Tests for src/datasets/common.py — the shared channel-order conversion used by every loader."""

import numpy as np
import pytest
import torch

from src.datasets.common import chw_to_hwc, hwc_to_chw_tensor, normalize_to_tanh_range


def test_chw_to_hwc_transposes_axes():
    # 2 channels, 3 rows, 4 columns — deliberately non-square so a transpose bug (e.g. swapping
    # H and W instead of moving C to the end) would show up as a shape mismatch, not just silently
    # produce a same-shaped-but-wrong array.
    chw = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    hwc = chw_to_hwc(chw)

    assert hwc.shape == (3, 4, 2)
    # Spot-check actual values move with the axes, not just the shape.
    assert np.array_equal(hwc[:, :, 0], chw[0])
    assert np.array_equal(hwc[:, :, 1], chw[1])


def test_chw_to_hwc_casts_to_float32():
    chw = np.zeros((2, 3, 4), dtype=np.uint16)
    hwc = chw_to_hwc(chw)
    assert hwc.dtype == np.float32


def test_chw_to_hwc_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        chw_to_hwc(np.zeros((3, 4)))  # missing the channel dimension


class TestNormalizeToTanhRange:
    def test_output_is_within_tanh_bounds(self):
        rng = np.random.default_rng(0)
        image = rng.normal(loc=500, scale=200, size=(16, 16, 2)).astype(np.float32)
        out = normalize_to_tanh_range(image)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_normalizes_each_channel_independently(self):
        # channel 0 has a much larger value range than channel 1 -- a shared low/high across
        # channels would let channel 0 dominate and wash out channel 1's stretch.
        image = np.zeros((10, 10, 2), dtype=np.float32)
        image[:, :, 0] = np.linspace(0, 1000, 100).reshape(10, 10)
        image[:, :, 1] = np.linspace(0, 1, 100).reshape(10, 10)
        out = normalize_to_tanh_range(image)
        assert out[:, :, 0].max() == pytest.approx(1.0, abs=1e-3)
        assert out[:, :, 1].max() == pytest.approx(1.0, abs=1e-3)

    def test_constant_channel_does_not_produce_nan(self):
        """A flat channel (high == low) would divide by zero without the epsilon guard."""
        image = np.full((8, 8, 1), 42.0, dtype=np.float32)
        out = normalize_to_tanh_range(image)
        assert not np.isnan(out).any()

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError):
            normalize_to_tanh_range(np.zeros((8, 8)))


class TestHwcToChwTensor:
    def test_transposes_and_returns_float_tensor(self):
        image = np.arange(2 * 3 * 4, dtype=np.float32).reshape(3, 4, 2)
        tensor = hwc_to_chw_tensor(image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 3, 4)
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor[0], torch.from_numpy(image[:, :, 0]))
