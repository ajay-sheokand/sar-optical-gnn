"""
The novel model this whole project exists to test (docs/RESEARCH_PLAN.md §6, M4): a GNN over the
SAR image's superpixel Region-Adjacency-Graph, injected as a structural prior into the same
pix2pix U-Net backbone the M3 baseline uses.

Architecture, matching §6's numbered steps:
    2-4. SLIC + RAG (src/graph_builder.py, M2) + per-node features (src/graph/features.py, M2)
         already give each node a (6 + 2*C) feature vector -- geometric shape descriptors plus
         per-channel SAR mean/std. `GraphBranch` runs GATConv layers over this graph.
    5. Node-level auxiliary supervision: the training script (scripts/train_gnn_hybrid.py) pools
       the *real* optical image into the same segmentation and supervises GraphBranch's output
       against it directly -- without this, the GNN has nothing forcing it to predict anything
       meaningful, since GATConv itself has no notion of "correct."
    6. Unpool (`unpool_torch`) broadcasts each node's prediction back onto every pixel in its
       superpixel -- a coarse, piecewise-constant "regional prior" image.
    7. That prior is concatenated onto the SAR input (extra channels) and the combined tensor is
       fed through `src.models.pix2pix.UNetGenerator` unchanged -- literally the same class M3's
       baseline uses, not a fork of it. This is what makes the ablation in §6 exact rather than
       approximate: `GraphHybridGenerator(..., use_graph_branch=False)` builds its wrapped
       UNetGenerator with `sar_channels` input channels instead of `sar_channels + optical_channels`,
       so disabling the graph branch doesn't just skip a computation at forward time, it produces
       the identical architecture (same parameter shapes) as a standalone pix2pix generator --
       checked directly in tests/models/test_gnn_hybrid.py, not just asserted in this docstring.

Batching: this first version processes one sample at a time (batch size 1 for the graph branch
specifically) rather than using torch_geometric's multi-graph Batch collation. Each SAR image has
its own differently-shaped graph (different node count, different edge_index), and properly
batching variable-sized graphs alongside dense image tensors is real additional complexity this
version defers rather than gets wrong under time pressure -- flagged here explicitly rather than
silently assumed, since "batch size 1" is a real performance cost (no parallelism across samples
for the graph branch) worth revisiting if training throughput becomes a bottleneck.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from src.models.pix2pix import UNetGenerator


def unpool_torch(node_predictions: torch.Tensor, label_map: torch.Tensor) -> torch.Tensor:
    """
    Differentiable node->pixel broadcast: every pixel gets its superpixel's predicted vector.
    The torch-autograd equivalent of src.graph.pooling.unpool, which is numpy-only (that module's
    job is offline feature caching, not a live, backprop-through-able model layer -- see its own
    docstring).

    Args:
        node_predictions: (N, C) -- one C-dim vector per node, N = number of superpixels. Requires
            grad if produced by a trainable branch (e.g. GraphBranch's output).
        label_map: (H, W) integer tensor with values in [0, N-1] -- *positional* node indices
            (row i of node_predictions), not raw superpixel label values. Callers are responsible
            for this remapping (src.datasets.graph_dataset.GraphHybridDataset does it once per
            sample at load time, not on every forward call).

    Returns:
        (C, H, W) float tensor, differentiable w.r.t. node_predictions.
    """
    if label_map.min() < 0 or label_map.max() >= node_predictions.shape[0]:
        raise ValueError(
            f"label_map values must be positional indices in [0, {node_predictions.shape[0] - 1}], "
            f"got range [{int(label_map.min())}, {int(label_map.max())}] -- did you forget to "
            f"remap raw superpixel label values to row indices first?"
        )
    # Fancy indexing with a (H, W) index tensor into an (N, C) tensor gives (H, W, C) directly;
    # advanced indexing is autograd-differentiable in torch, so gradients flow back into
    # node_predictions without needing a custom backward pass.
    return node_predictions[label_map].permute(2, 0, 1)


class GraphBranch(nn.Module):
    """
    GATConv stack over the SAR superpixel graph. Input: one node-feature matrix (N, node_feature_dim)
    for a single sample's graph (see this module's docstring on why batch size is 1 here). Output:
    (N, optical_channels) -- a predicted optical value per node, bounded to tanh's [-1, 1] range to
    match every other tensor in this project's pipeline (src/datasets/common.py's
    normalize_to_tanh_range), since its target (real optical, pooled per-node) is in that range too.
    """

    def __init__(self, node_feature_dim: int, hidden_dim: int, optical_channels: int, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers = []
        in_dim = node_feature_dim
        for _ in range(num_layers - 1):
            layers.append(GATConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        layers.append(GATConv(in_dim, optical_channels))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ELU()  # standard GAT-paper choice between layers

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = node_features
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return torch.tanh(x)


class GraphHybridGenerator(nn.Module):
    """
    The novel model. Wraps src.models.pix2pix.UNetGenerator, optionally prepending a graph-based
    structural prior to its input channels -- see this module's docstring for the exact ablation
    property this construction gives.
    """

    def __init__(
        self,
        sar_channels: int,
        optical_channels: int,
        node_feature_dim: int,
        num_downs: int = 6,
        gnn_hidden_dim: int = 64,
        gnn_layers: int = 2,
        use_graph_branch: bool = True,
    ):
        super().__init__()
        self.use_graph_branch = use_graph_branch

        if use_graph_branch:
            self.graph_branch = GraphBranch(node_feature_dim, gnn_hidden_dim, optical_channels, gnn_layers)
            unet_in_channels = sar_channels + optical_channels
        else:
            self.graph_branch = None
            unet_in_channels = sar_channels

        self.unet = UNetGenerator(unet_in_channels, optical_channels, num_downs=num_downs)

    def forward(self, sar, node_features=None, edge_index=None, label_map=None):
        """
        Args:
            sar: (1, sar_channels, H, W) -- batch size 1, see module docstring.
            node_features: (N, node_feature_dim), required if use_graph_branch.
            edge_index: (2, E) torch_geometric edge index, required if use_graph_branch.
            label_map: (H, W) positional node-index tensor (see unpool_torch), required if
                use_graph_branch.

        Returns:
            (fake_optical, node_predictions) -- node_predictions is None when the graph branch is
            disabled (nothing to supervise with a node-aux loss in that case).
        """
        if not self.use_graph_branch:
            return self.unet(sar), None

        if node_features is None or edge_index is None or label_map is None:
            raise ValueError("node_features, edge_index, and label_map are all required when use_graph_branch=True")

        node_predictions = self.graph_branch(node_features, edge_index)
        prior = unpool_torch(node_predictions, label_map).unsqueeze(0)  # (1, optical_channels, H, W)
        combined = torch.cat([sar, prior], dim=1)
        return self.unet(combined), node_predictions
