"""Tests for src/datasets/common.py — the shared channel-order conversion used by every loader."""

import numpy as np
import pytest

from src.datasets.common import chw_to_hwc


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
