"""
Tests for src/datasets/bigearthnet.py's sample-conversion logic.

We deliberately do NOT test against the real BigEarthNet dataset here — it's hundreds of GB and
requires a deliberate `download=True` step, not something a test suite should trigger. Instead we
test `_convert_sample`, the pure function that does this project's actual logic (torchgeo's own
BigEarthNetV2 class already has its own upstream test coverage for the parts that touch real
files — re-testing that isn't this project's job).
"""

import numpy as np
import torch

from src.datasets.bigearthnet import _convert_sample


def _fake_raw_sample():
    """Build a sample shaped exactly like torchgeo.datasets.BigEarthNetV2(bands='all')[i]."""
    return {
        "image_s1": torch.zeros(2, 120, 120),
        "image_s2": torch.arange(12 * 120 * 120, dtype=torch.float32).reshape(12, 120, 120),
        "mask": torch.zeros(1, 120, 120, dtype=torch.long),
        "label": torch.zeros(19, dtype=torch.long),
    }


def test_convert_sample_shapes():
    converted = _convert_sample(_fake_raw_sample())

    assert converted["sar"].shape == (120, 120, 2)
    assert converted["optical"].shape == (120, 120, 12)
    assert converted["mask"].shape == (120, 120)  # leading singleton band dim squeezed out
    assert converted["label"].shape == (19,)


def test_convert_sample_preserves_values():
    raw = _fake_raw_sample()
    converted = _convert_sample(raw)

    # channel 0 of optical, at pixel (0, 0), should be the same number before and after the
    # channel-order transpose -- catches a transpose bug that reshuffles values, not just axes.
    assert converted["optical"][0, 0, 0] == raw["image_s2"][0, 0, 0].item()


def test_convert_sample_mask_dtype_is_int():
    converted = _convert_sample(_fake_raw_sample())
    assert np.issubdtype(converted["mask"].dtype, np.integer)
