"""
Wraps a base SAR/optical dataset (src/datasets/*.py) plus this project's offline graph cache
(scripts/build_graphs_offline.py, M2) into exactly what src.models.gnn_hybrid.GraphHybridGenerator
and its node-auxiliary loss (docs/RESEARCH_PLAN.md §6 step 5) need per sample.

Three things this module does that aren't just "load and concatenate":
  1. Remaps `labels`' raw superpixel label values to *positional* row indices into the cached
     feature matrix -- see src.models.gnn_hybrid.unpool_torch's docstring for why something has
     to do this remapping, and why doing it once here (not on every forward() call) is cheaper.
  2. Precomputes the node-auxiliary loss's target: pools the *real* optical image into the same
     segmentation the SAR-derived graph used, giving one target vector per node in the same row
     order as the cached feature matrix -- ready for a direct L1 comparison against
     GraphBranch's output, no per-training-step numpy round-trip needed.
  3. Normalizes the cached node feature matrix (`normalize_node_features`) -- found necessary by
     an actual failed test, not added speculatively: src.graph.features.compute_node_features
     returns raw-scale values (pixel-count `area`, pixel-coordinate `centroid`, raw channel
     mean/std) with wildly different magnitudes -- area alone can be in the hundreds. Fed
     unnormalized into GATConv, activations reached +-200 after two layers, which saturates
     GraphBranch's final `tanh` completely (output pinned to exactly +-1.0), and a saturated tanh
     has *zero* derivative -- every single graph_branch parameter got exactly zero gradient,
     confirmed directly (not assumed) by comparing parameters before/after a real training step.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.utils.data

from scripts.build_graphs_offline import load_cached_graph
from src.datasets.common import hwc_to_chw_tensor, normalize_to_tanh_range
from src.graph.pooling import scatter_pool


def normalize_node_features(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Per-column (per-feature-dimension) z-score normalization, computed from this one sample's own
    node population (mean/std across its N nodes) -- not a global dataset statistic. Consistent
    with this project's existing per-sample normalization precedent (src/datasets/common.py's
    normalize_to_tanh_range does the same thing for images, for the same reason: simpler, and
    doesn't need a separate statistics-computation pass over the whole dataset before training
    can start).

    Known limitation, not silently assumed away: a sample with very few nodes gives a noisy std
    estimate, and in the degenerate N=1 case every column's std is exactly 0 (no variance across
    a single node), collapsing all its features to 0 after the epsilon guard below -- numerically
    safe (no NaN/inf), but loses that sample's actual feature information. Real segmentations
    (this project's default num_segments=100) produce far more nodes than that; revisit with
    global dataset statistics if per-sample instability shows up in practice.
    """
    mean = feature_matrix.mean(axis=0, keepdims=True)
    std = feature_matrix.std(axis=0, keepdims=True)
    return ((feature_matrix - mean) / (std + 1e-6)).astype(np.float32)


class GraphHybridDataset(torch.utils.data.Dataset):
    """
    Args:
        base_dataset: anything with __len__/__getitem__ returning {"sar", "optical"} (H, W, C)
            numpy dicts (src/datasets/common.py's shared contract).
        graph_cache_dir: directory of `{index:07d}.npz` files produced by
            scripts/build_graphs_offline.py's cache_graph() for this same base_dataset, in the
            same index order. Not verified against the base dataset's actual contents here --
            a mismatched cache would silently pair the wrong graph with the wrong image, which is
            why scripts/train_gnn_hybrid.py caches graphs immediately before training rather than
            trusting a possibly-stale cache from a different dataset/version.
    """

    def __init__(self, base_dataset, graph_cache_dir):
        self.base_dataset = base_dataset
        self.graph_cache_dir = graph_cache_dir

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        cache_path = os.path.join(self.graph_cache_dir, f"{index:07d}.npz")
        graph = load_cached_graph(cache_path)

        labels = graph["labels"]
        node_ids = graph["node_ids"]
        # node_ids is sorted (graph_to_arrays' own contract) and every value in `labels` is
        # guaranteed present in node_ids by construction, so searchsorted gives the exact
        # position of each pixel's label within the feature matrix's row order.
        label_map = np.searchsorted(node_ids, labels).astype(np.int64)

        optical_normalized = normalize_to_tanh_range(sample["optical"])
        node_target_dict = scatter_pool(optical_normalized, labels, reduction="mean")
        node_targets = np.stack([node_target_dict[int(node_id)] for node_id in node_ids]).astype(np.float32)

        return {
            "sar": hwc_to_chw_tensor(normalize_to_tanh_range(sample["sar"])),
            "optical": hwc_to_chw_tensor(optical_normalized),
            "node_features": torch.from_numpy(normalize_node_features(graph["feature_matrix"])).float(),
            "edge_index": torch.from_numpy(graph["edge_index"]).long(),
            "label_map": torch.from_numpy(label_map),
            "node_targets": torch.from_numpy(node_targets),
        }
