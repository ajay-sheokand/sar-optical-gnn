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
import torch


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


def normalize_to_tanh_range(image, low_percentile=2.0, high_percentile=98.0) -> np.ndarray:
    """
    Per-channel percentile stretch an (H, W, C) array into [-1, 1] -- the range every generator
    in src/models/ produces via its final Tanh (see src/models/pix2pix.py, src/models/cyclegan.py)
    and the range src/eval/metrics.py's TranslationMetrics assumes.

    Percentile (not true min/max) stretch, independently per channel, for the same reason
    scripts/visualize_sample.py's _normalize_for_display uses it: SAR backscatter in particular
    tends to have a handful of extreme outlier pixels that would wash out the rest of the dynamic
    range under a true min/max normalization. Per-channel (not per-image) because SAR's VV and VH
    polarizations, and optical's individual spectral bands, have genuinely different value
    distributions from each other -- a single shared low/high would let one channel dominate.

    This is a per-sample normalization, not a dataset-wide statistic (e.g. a precomputed mean/std)
    -- simpler to get right first, and avoids a training-time dependency on a normalization pass
    over the whole dataset before training can start. Revisit if per-sample percentile noise
    turns out to hurt training stability (an open question, not yet observed).
    """
    if image.ndim != 3:
        raise ValueError(f"expected a 3D (H, W, C) array, got shape {image.shape}")

    image = image.astype(np.float32)
    low = np.percentile(image, low_percentile, axis=(0, 1), keepdims=True)
    high = np.percentile(image, high_percentile, axis=(0, 1), keepdims=True)
    # A fixed 1e-6 epsilon silently fails for a constant channel with a large-magnitude value
    # (e.g. 42.0 + 1e-6 rounds back to exactly 42.0 in float32, since that exceeds float32's ~7
    # significant digits) -- found by test_constant_channel_does_not_produce_nan actually failing
    # with this fixed-epsilon version. Scaling the epsilon to the value's own magnitude keeps it
    # representable regardless of how large low/high are.
    eps = np.maximum(np.abs(low) * 1e-6, 1e-6)
    high = np.where(high <= low, low + eps, high)
    stretched = np.clip((image - low) / (high - low), 0, 1)
    return (stretched * 2 - 1).astype(np.float32)


def hwc_to_chw_tensor(image):
    """Convert an (H, W, C) numpy array to a (C, H, W) float32 torch.Tensor -- the layout every
    src/models/ generator and src/eval/metrics.py's TranslationMetrics expect."""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
