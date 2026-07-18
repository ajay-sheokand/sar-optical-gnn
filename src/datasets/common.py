"""
Shared conventions for every dataset loader in this package.

Why this file exists: BigEarthNet, SEN12MS, and SEN1-2 each ship data in a different on-disk
layout, and torchgeo (used for the first two) returns samples as channel-first (C, H, W) torch
tensors — the PyTorch convention. But src/graph_builder.py's SLIC segmentation and RAG
construction (skimage functions) expect channel-*last* (H, W, C) numpy arrays — the image
processing convention. Rather than repeat that conversion, and the reasoning behind it, in three
separate files, it lives here once.

Every dataset wrapper in this package returns a plain dict with the same three keys:
    {"sar": (H, W, C_sar) float32 array, "optical": (H, W, C_optical) float32 array, ...}
plus whatever extra per-dataset fields make sense (a land-cover mask, a multi-label vector).
This shared shape is what src/graph/pooling.py (M2) and the model code (M3+) are written against,
so a training loop doesn't need to know which of the three source datasets a sample came from.
"""

from __future__ import annotations

import numpy as np


def chw_to_hwc(array) -> np.ndarray:
    """
    Convert a channel-first (C, H, W) array/tensor to channel-last (H, W, C) float32 numpy.

    Accepts either a torch.Tensor (as returned by torchgeo dataset classes) or a numpy array (as
    returned by rasterio.read()) — both use the same (C, H, W) axis order, so both are handled by
    the same `np.asarray(...).transpose(1, 2, 0)` regardless of which library produced them.
    """
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"expected a 3D (C, H, W) array, got shape {array.shape}")
    return array.transpose(1, 2, 0).astype(np.float32)
