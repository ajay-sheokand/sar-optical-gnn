"""
Per-node feature vectors for the GNN: geometric shape descriptors (via skimage.measure.regionprops)
concatenated with per-channel mean/std statistics (via src.graph.pooling) — the "zonal statistics"
step from docs/UNDERSTANDING_THE_PROJECT.md §4 step 3.

Why these two families of features specifically (docs/RESEARCH_PLAN.md §6 step 3): a superpixel's
*shape* (is it long and thin, like a river or road? compact, like a field?) carries information a
pure color/backscatter average throws away — two regions with identical mean backscatter can still
be very different land-cover types depending on their shape. `regionprops` gives us that for free
from the label map alone, no image content needed.

This module doesn't yet include CNN-pooled features (docs/RESEARCH_PLAN.md §6 step 3's third
ingredient) — there's no CNN encoder until M3/M4. `compute_node_features` is written so that adding
them later is a matter of pooling an extra feature map with `src.graph.pooling.scatter_pool` and
concatenating the result, not a redesign.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import regionprops

from src.graph.pooling import scatter_pool

#: Which skimage regionprops properties to use, and in what order — fixed here so the resulting
#: feature vector's layout is documented in one place rather than implied by dict-iteration order.
#: `area`, `eccentricity`, `orientation`, `extent` are all scale-appropriate for the small
#: (~100-1000px) superpixel regions this project's segmentation produces; deliberately excluding
#: shape descriptors like `perimeter` that are more sensitive to a superpixel's typically-jagged
#: boundary (an artifact of SLIC's pixel-grid segmentation, not a meaningful shape signal).
_GEOMETRIC_PROPERTIES = ("area", "eccentricity", "orientation", "extent")


def compute_geometric_features(labels: np.ndarray) -> dict[int, np.ndarray]:
    """
    Per-superpixel shape descriptors from the label map alone (no image content needed).

    Returns:
        {label: (4,) float32 array} — [area, eccentricity, orientation, extent], per
        _GEOMETRIC_PROPERTIES, plus the (row, col) centroid appended, so (6,) total per node.
        One entry per label present in `labels`.
    """
    features: dict[int, np.ndarray] = {}
    for region in regionprops(labels):
        values = [getattr(region, name) for name in _GEOMETRIC_PROPERTIES]
        values.extend(region.centroid)  # (row, col), appended last
        features[int(region.label)] = np.array(values, dtype=np.float32)
    return features


def compute_channel_statistics(image_data: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """
    Per-superpixel mean AND standard deviation, per channel — richer than
    src.graph.pooling.scatter_pool's mean-only output, at the cost of being its own pass. Std is
    specifically relevant for SAR: mean backscatter alone doesn't distinguish a smooth surface
    from a rough one with the same average return, but variance within a region does.

    Returns:
        {label: (2*C,) float32 array} — [mean_0..mean_{C-1}, std_0..std_{C-1}].
    """
    means = scatter_pool(image_data, labels, reduction="mean")

    stds: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        mask = labels == label
        stds[int(label)] = image_data[mask].std(axis=0).astype(np.float32)

    return {label: np.concatenate([means[label], stds[label]]) for label in means}


def compute_node_features(image_data: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """
    The full per-node feature vector this project's GNN (M4) will consume: geometric descriptors
    concatenated with per-channel mean/std statistics.

    Returns:
        {label: (6 + 2*C,) float32 array}, one entry per superpixel.
    """
    geometric = compute_geometric_features(labels)
    channel_stats = compute_channel_statistics(image_data, labels)

    return {
        label: np.concatenate([geometric[label], channel_stats[label]])
        for label in geometric
    }
