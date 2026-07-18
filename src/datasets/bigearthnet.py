"""
BigEarthNet v2.0 (reBEN) loader — the project's primary experimental dataset.

Why BigEarthNet specifically (see docs/RESEARCH_PLAN.md §4 for the full comparison against SEN1-2
and SEN12MS): it's the only candidate dataset with paired Sentinel-1/Sentinel-2 imagery *and* real,
non-proxy CORINE Land Cover labels *and* an established, versioned benchmark. That combination is
required for two different jobs later in the pipeline: training the GNN-hybrid translation model
(needs paired SAR/optical), and the downstream classification-fidelity check in M5 (needs real
labels to train a classifier on).

This module doesn't reimplement dataset parsing — torchgeo.datasets.BigEarthNetV2 already handles
the real on-disk layout (tarball extraction, patch-ID matching between S1/S2/label-map files,
verification/download). What this module adds is a thin wrapper that:
  1. Fixes the options this project always wants (bands='all', so every sample has both SAR and
     optical available; num_classes is fixed at 19 in V2, unlike V1's 19-or-43 choice).
  2. Converts torchgeo's channel-first torch tensors into the channel-last numpy arrays that
     src/graph_builder.py's SLIC/RAG functions expect (see src/datasets/common.py).
  3. Renames torchgeo's `image_s1`/`image_s2` keys to this project's shared `sar`/`optical`
     convention, so downstream code doesn't need to know which dataset a sample came from.

Downloading the real dataset (~549K patches) is NOT done as part of writing this module — it's a
large, multi-part download (S1 + S2 + reference maps + metadata, several GB minimum even for a
useful subset) gated behind `download=True`, left for whoever actually runs training to trigger
deliberately. See _convert_sample's docstring for how this file's logic is tested without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torchgeo.datasets import BigEarthNetV2

from src.datasets.common import chw_to_hwc


def _convert_sample(raw_sample: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one torchgeo BigEarthNetV2 sample into this project's shared sar/optical convention.

    Pulled out as a standalone function (rather than inlined in __getitem__) specifically so it
    can be unit-tested against a hand-built fake sample dict, without needing the real,
    multi-gigabyte BigEarthNet dataset on disk — see tests/datasets/test_bigearthnet.py.

    Args:
        raw_sample: a dict as returned by BigEarthNetV2(bands="all").__getitem__, i.e.
            {"image_s1": (2, 120, 120) tensor, "image_s2": (12, 120, 120) tensor,
             "mask": (1, 120, 120) tensor of ordinal CORINE class indices,
             "label": (19,) multi-hot tensor}

    Returns:
        {"sar": (120, 120, 2) float32 array, "optical": (120, 120, 12) float32 array,
         "mask": (120, 120) int array, "label": (19,) int array}
    """
    mask = np.asarray(raw_sample["mask"])
    # torchgeo stores the reference map with a leading singleton band dimension (1, H, W);
    # squeeze it down to (H, W) since it's a per-pixel class map, not a multi-channel image.
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    return {
        "sar": chw_to_hwc(raw_sample["image_s1"]),
        "optical": chw_to_hwc(raw_sample["image_s2"]),
        "mask": mask.astype(np.int64),
        "label": np.asarray(raw_sample["label"]).astype(np.int64),
    }


class BigEarthNetSAROptical:
    """
    Thin, project-convention wrapper around torchgeo.datasets.BigEarthNetV2.

    Usage:
        ds = BigEarthNetSAROptical(root="data/bigearthnet", split="train", download=True)
        sample = ds[0]
        sample["sar"].shape       # (120, 120, 2)  -- VV, VH
        sample["optical"].shape   # (120, 120, 12) -- 12 Sentinel-2 bands
        sample["mask"].shape      # (120, 120)     -- per-pixel CORINE class (19-class ordinal)
        sample["label"].shape     # (19,)          -- multi-hot scene-level label
    """

    #: The 19 CORINE-derived class names, in the same order the ordinal indices in `mask` and
    #: `label` use. Exposed here so downstream code (e.g. the M5 classifier) doesn't have to reach
    #: into torchgeo internals to get human-readable class names.
    class_names = BigEarthNetV2.class_set

    def __init__(
        self,
        root: str = "data/bigearthnet",
        split: str = "train",
        download: bool = False,
    ) -> None:
        self._dataset = BigEarthNetV2(
            root=root,
            split=split,
            bands="all",
            download=download,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _convert_sample(self._dataset[index])
