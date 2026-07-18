"""
Tests for src/datasets/sen12ms.py's sample-conversion logic.

Same philosophy as tests/datasets/test_bigearthnet.py: no real ~85GB SEN12MS download involved,
just the pure function that splits torchgeo's single concatenated 15-channel tensor into this
project's separate sar/optical convention.
"""

import numpy as np
import torch

from src.datasets.sen12ms import _convert_sample


def _fake_raw_sample():
    """
    Build a sample shaped like torchgeo.datasets.SEN12MS(bands=SEN12MS.BAND_SETS['all'])[i]:
    channels 0-1 are SAR (VV, VH), channels 2-14 are the 13 Sentinel-2 optical bands — this
    specific ordering is what _convert_sample assumes, per torchgeo's own documented band order.
    """
    image = torch.arange(15 * 256 * 256, dtype=torch.float32).reshape(15, 256, 256)
    mask = torch.zeros(256, 256, dtype=torch.long)
    return {"image": image, "mask": mask}


def test_convert_sample_splits_sar_and_optical():
    converted = _convert_sample(_fake_raw_sample())

    assert converted["sar"].shape == (256, 256, 2)
    assert converted["optical"].shape == (256, 256, 13)
    assert converted["mask"].shape == (256, 256)


def test_convert_sample_splits_at_the_right_channel():
    raw = _fake_raw_sample()
    converted = _convert_sample(raw)

    # The first SAR channel (index 0 of the original 15) must land in sar[..., 0], and the first
    # optical channel (index 2 of the original 15) must land in optical[..., 0] -- not shifted by
    # one, which is the easy off-by-one mistake with a 2-channel-then-13-channel split.
    assert converted["sar"][0, 0, 0] == raw["image"][0, 0, 0].item()
    assert converted["optical"][0, 0, 0] == raw["image"][2, 0, 0].item()


def test_convert_sample_mask_dtype_is_int():
    converted = _convert_sample(_fake_raw_sample())
    assert np.issubdtype(converted["mask"].dtype, np.integer)
