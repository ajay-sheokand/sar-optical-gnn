"""
Pixel <-> superpixel-node pooling: the two operations that move data between the pixel grid and
the graph, in both directions.

Where this fits in the architecture (docs/RESEARCH_PLAN.md §6, docs/UNDERSTANDING_THE_PROJECT.md
§4): step 3 ("superpixel pooling — the deep-learning equivalent of zonal statistics") is
`scatter_pool` below; step 6 ("unpooling — scatter refined node embeddings back to the pixel
grid") is `unpool` below. Both are written generically over an arbitrary per-pixel feature map
(any number of channels), not specifically over a 3-channel image — this is a direct consequence
of the bug found while fixing src/graph_builder.py: skimage's `rag_mean_color` hardcodes a
3-channel accumulator, which is one of the two reasons this project stopped depending on it.
graph_builder.py's own per-node mean-color computation is `scatter_pool(image, labels, reduction="mean")`
under the hood, so there's exactly one implementation of "average a feature map over each
superpixel," not two.

A note on performance, since docs/RESEARCH_PLAN.md §7 explicitly calls out benchmarking graph
construction throughput early rather than assuming it's fine: `scatter_pool` with `reduction="mean"`
is fully vectorized (one `np.add.at` scatter-add over every pixel, no Python-level loop over
superpixels) — this matters because the original `rag_mean_color` accumulates one pixel at a time
via `np.ndindex`, a pure-Python loop over every pixel in the image. `reduction="max"` cannot be
vectorized the same way (no numpy scatter-max primitive) and falls back to one loop iteration per
superpixel, which is still far fewer iterations than one per pixel.
"""

from __future__ import annotations

import numpy as np


def scatter_pool(
    feature_map: np.ndarray, labels: np.ndarray, reduction: str = "mean"
) -> dict[int, np.ndarray]:
    """
    Pool a per-pixel feature map into one feature vector per superpixel (pixel -> node).

    Args:
        feature_map: (H, W, C) array — any number of channels.
        labels: (H, W) integer array of superpixel labels, e.g. from
            `src.graph_builder.build_graph_from_image`. Must have the same (H, W) as feature_map.
        reduction: "mean" or "max" — how to combine the pixels within a superpixel into one
            vector. "mean" is the default used for RAG edge weights; "max" is the granularity/
            node-feature ablation option from docs/RESEARCH_PLAN.md §6.

    Returns:
        {label: (C,) float32 array}, one entry per unique value in `labels`.
    """
    if feature_map.shape[:2] != labels.shape:
        raise ValueError(
            f"feature_map spatial shape {feature_map.shape[:2]} does not match "
            f"labels shape {labels.shape}"
        )
    if reduction not in ("mean", "max"):
        raise ValueError(f"reduction must be 'mean' or 'max', got {reduction!r}")

    if reduction == "mean":
        return _scatter_mean(feature_map, labels)
    return _scatter_max(feature_map, labels)


def _scatter_mean(feature_map: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    height, width, num_channels = feature_map.shape
    flat_features = feature_map.reshape(-1, num_channels)
    flat_labels = labels.reshape(-1)

    unique_labels, inverse_indices = np.unique(flat_labels, return_inverse=True)
    num_nodes = len(unique_labels)

    sums = np.zeros((num_nodes, num_channels), dtype=np.float64)
    # Scatter-add every pixel's feature vector into its node's running sum in one vectorized
    # pass -- this is the operation a plain `for pixel in image: ...` loop would otherwise need,
    # done here without leaving numpy.
    np.add.at(sums, inverse_indices, flat_features)
    counts = np.bincount(inverse_indices, minlength=num_nodes)

    means = (sums / counts[:, None]).astype(np.float32)
    return {int(label): means[i] for i, label in enumerate(unique_labels)}


def _scatter_max(feature_map: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        mask = labels == label
        result[int(label)] = feature_map[mask].max(axis=0).astype(np.float32)
    return result


def unpool(node_features: dict[int, np.ndarray], labels: np.ndarray) -> np.ndarray:
    """
    Broadcast per-node feature vectors back onto the pixel grid (node -> pixel) — every pixel in
    a given superpixel gets that superpixel's vector. Produces the coarse, piecewise-constant
    "regional prior" map described in docs/RESEARCH_PLAN.md §6 step 6, which later gets
    concatenated with per-pixel CNN features and decoded (M4).

    Args:
        node_features: {label: (C,) array} — e.g. the output of scatter_pool, or a GNN's refined
            per-node embeddings once the model exists (M4).
        labels: (H, W) integer array of superpixel labels, matching the keys of `node_features`.

    Returns:
        (H, W, C) float32 array.
    """
    any_vector = next(iter(node_features.values()))
    num_channels = any_vector.shape[0]
    height, width = labels.shape

    output = np.zeros((height, width, num_channels), dtype=np.float32)
    for label, vector in node_features.items():
        output[labels == label] = vector
    return output


def get_node_mask(labels: np.ndarray, node_id: int) -> np.ndarray:
    """
    The node<->pixel-mask helper called for in docs/RESEARCH_PLAN.md's repo-structure notes:
    given a label map and one node's id, return the boolean pixel mask of that superpixel's
    region. Trivial (`labels == node_id`) but named and tested explicitly so callers (e.g. M2's
    features.py, or plotting code) have one obvious place to get this instead of re-deriving the
    same one-liner inconsistently in several files.
    """
    return labels == node_id
