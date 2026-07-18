"""
SEN12MS loader — this project's secondary dataset (docs/RESEARCH_PLAN.md §4).

Used specifically for two jobs that need larger patches than BigEarthNet's 120x120px provides:
  1. The superpixel-granularity (MAUP) ablation — 256x256px patches give more room to vary
     `num_segments` meaningfully than 120x120px does.
  2. Cross-dataset generalization testing — train on BigEarthNet, zero-shot evaluate here.

Its MODIS-derived land-cover labels are too coarse (~500m resolution, reprojected down to 10m) to
anchor the downstream classification-fidelity check (M5) — that job stays with BigEarthNet, which
has real CORINE labels.

Like bigearthnet.py, this wraps torchgeo.datasets.SEN12MS rather than re-parsing the dataset's
on-disk layout. torchgeo's SEN12MS concatenates SAR and optical bands into a single `image` tensor
(see its docstring: "Indices 0 and 1 correspond to Sentinel-1, indices 2 through 14 correspond to
Sentinel-2") — this wrapper always requests the *full*, fixed band order (never a caller-chosen
subset) specifically so that "first 2 channels = SAR, remaining 13 = optical" is guaranteed true by
construction, not by hoping the caller didn't reorder anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torchgeo.datasets import SEN12MS

from src.datasets.common import chw_to_hwc

#: SEN12MS's SAR channels (VV, VH) always occupy the first 2 positions of the "all" band set;
#: the remaining 13 positions are the Sentinel-2 optical bands. This split point is a property of
#: torchgeo's fixed BAND_SETS['all'] ordering, not something we're free to change independently.
_NUM_SAR_CHANNELS = 2


def _convert_sample(raw_sample: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one torchgeo SEN12MS sample (all 15 bands, fixed order) into the sar/optical split.

    Pulled out as a standalone function so it can be unit-tested against a hand-built sample dict
    without needing the real ~85GB SEN12MS download — see tests/datasets/test_sen12ms.py.

    Args:
        raw_sample: {"image": (15, 256, 256) tensor, "mask": (256, 256) tensor of IGBP land-cover
            class indices}

    Returns:
        {"sar": (256, 256, 2) float32 array, "optical": (256, 256, 13) float32 array,
         "mask": (256, 256) int array}
    """
    image = np.asarray(raw_sample["image"])
    sar_chw = image[:_NUM_SAR_CHANNELS]
    optical_chw = image[_NUM_SAR_CHANNELS:]

    return {
        "sar": chw_to_hwc(sar_chw),
        "optical": chw_to_hwc(optical_chw),
        "mask": np.asarray(raw_sample["mask"]).astype(np.int64),
    }


class SEN12MSSAROptical:
    """
    Thin, project-convention wrapper around torchgeo.datasets.SEN12MS.

    Usage:
        ds = SEN12MSSAROptical(root="data/sen12ms", split="train")
        sample = ds[0]
        sample["sar"].shape       # (256, 256, 2)  -- VV, VH
        sample["optical"].shape   # (256, 256, 13) -- 13 Sentinel-2 bands
        sample["mask"].shape      # (256, 256)     -- MODIS IGBP land-cover class (coarse, see
                                   #                  module docstring for why this isn't used for
                                   #                  the downstream classification check)

    Note: torchgeo's SEN12MS has no `download=True` option (unlike BigEarthNetV2) — SEN12MS is
    distributed via an FTP mirror that must be fetched manually first; see torchgeo's docstring
    for the exact wget/tar commands, or docs/RESEARCH_PLAN.md §4 for the access pointer.
    """

    def __init__(self, root: str = "data/sen12ms", split: str = "train") -> None:
        self._dataset = SEN12MS(
            root=root,
            split=split,
            bands=SEN12MS.BAND_SETS["all"],
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _convert_sample(self._dataset[index])
