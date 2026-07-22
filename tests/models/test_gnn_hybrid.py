"""
Tests for src/models/gnn_hybrid.py -- the novel model (M4). Beyond the usual shape/gradient
checks, the property that actually matters for this project's research claim gets its own test:
disabling the graph branch must produce the *exact same architecture* as M3's pix2pix baseline,
not just "skip a computation" -- docs/RESEARCH_PLAN.md §6 calls this out explicitly as what makes
the ablation meaningful.
"""

import pytest
import torch

from src.models.gnn_hybrid import GraphBranch, GraphHybridGenerator, unpool_torch
from src.models.pix2pix import UNetGenerator


def _make_small_graph(num_nodes=6, feature_dim=10):
    """A small connected-ish graph -- both edge directions included, matching
    scripts/build_graphs_offline.py's graph_to_arrays convention (undirected via explicit
    duplication, not an undirected-aware conv layer)."""
    node_features = torch.randn(num_nodes, feature_dim)
    directed_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    edges = directed_edges + [(v, u) for u, v in directed_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    return node_features, edge_index


def _make_label_map(height, width, num_nodes):
    """A (H, W) positional-index label map covering all num_nodes at least once."""
    labels = torch.arange(height * width, dtype=torch.long) % num_nodes
    return labels.reshape(height, width)


class TestUnpoolTorch:
    def test_broadcasts_each_nodes_vector_to_its_pixels(self):
        node_predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        label_map = torch.tensor([[0, 1], [1, 0]])

        out = unpool_torch(node_predictions, label_map)

        assert out.shape == (2, 2, 2)
        assert torch.equal(out[:, 0, 0], torch.tensor([1.0, 2.0]))
        assert torch.equal(out[:, 0, 1], torch.tensor([3.0, 4.0]))
        assert torch.equal(out[:, 1, 1], torch.tensor([1.0, 2.0]))

    def test_is_differentiable_back_into_node_predictions(self):
        node_predictions = torch.randn(4, 3, requires_grad=True)
        label_map = torch.tensor([[0, 1], [2, 3]])

        out = unpool_torch(node_predictions, label_map)
        out.sum().backward()

        assert node_predictions.grad is not None
        assert (node_predictions.grad != 0).all()

    def test_rejects_out_of_range_label_map(self):
        node_predictions = torch.randn(3, 2)
        label_map = torch.tensor([[0, 1], [2, 3]])  # 3 is out of range for 3 nodes (valid: 0-2)

        with pytest.raises(ValueError, match="positional indices"):
            unpool_torch(node_predictions, label_map)


class TestGraphBranch:
    def test_output_shape_and_tanh_bounds(self):
        node_features, edge_index = _make_small_graph(num_nodes=6, feature_dim=10)
        branch = GraphBranch(node_feature_dim=10, hidden_dim=16, optical_channels=3, num_layers=2)

        out = branch(node_features, edge_index)

        assert out.shape == (6, 3)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_gradients_flow_to_every_layer(self):
        node_features, edge_index = _make_small_graph()
        branch = GraphBranch(node_feature_dim=10, hidden_dim=16, optical_channels=3, num_layers=3)

        out = branch(node_features, edge_index)
        out.sum().backward()

        for name, param in branch.named_parameters():
            assert param.grad is not None, f"no gradient reached {name}"

    def test_rejects_zero_layers(self):
        with pytest.raises(ValueError, match="num_layers"):
            GraphBranch(node_feature_dim=10, hidden_dim=16, optical_channels=3, num_layers=0)


class TestGraphHybridGenerator:
    def test_forward_with_graph_branch_returns_both_outputs(self):
        node_features, edge_index = _make_small_graph(num_nodes=6, feature_dim=10)
        label_map = _make_label_map(32, 32, num_nodes=6)
        sar = torch.randn(1, 2, 32, 32)

        model = GraphHybridGenerator(
            sar_channels=2, optical_channels=3, node_feature_dim=10, num_downs=3, use_graph_branch=True
        )
        fake_optical, node_predictions = model(sar, node_features, edge_index, label_map)

        assert fake_optical.shape == (1, 3, 32, 32)
        assert node_predictions.shape == (6, 3)

    def test_gradients_flow_to_both_graph_branch_and_unet(self):
        node_features, edge_index = _make_small_graph(num_nodes=6, feature_dim=10)
        label_map = _make_label_map(32, 32, num_nodes=6)
        sar = torch.randn(1, 2, 32, 32)

        model = GraphHybridGenerator(
            sar_channels=2, optical_channels=3, node_feature_dim=10, num_downs=3, use_graph_branch=True
        )
        fake_optical, node_predictions = model(sar, node_features, edge_index, label_map)
        (fake_optical.sum() + node_predictions.sum()).backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"no gradient reached {name}"

    def test_forward_without_graph_branch_needs_only_sar(self):
        sar = torch.randn(1, 2, 32, 32)
        model = GraphHybridGenerator(
            sar_channels=2, optical_channels=3, node_feature_dim=10, num_downs=3, use_graph_branch=False
        )

        fake_optical, node_predictions = model(sar)

        assert fake_optical.shape == (1, 3, 32, 32)
        assert node_predictions is None

    def test_raises_if_graph_branch_enabled_but_graph_args_missing(self):
        sar = torch.randn(1, 2, 32, 32)
        model = GraphHybridGenerator(
            sar_channels=2, optical_channels=3, node_feature_dim=10, num_downs=3, use_graph_branch=True
        )
        with pytest.raises(ValueError, match="required"):
            model(sar)

    def test_disabling_graph_branch_collapses_to_the_exact_pix2pix_architecture(self):
        """The actual research-methodology claim (docs/RESEARCH_PLAN.md §6): removing the graph
        branch must produce the *identical* architecture to a standalone pix2pix generator, not
        merely skip a computation at forward time -- checked by comparing every parameter's shape
        between the two, not just that both happen to run without error."""
        hybrid = GraphHybridGenerator(
            sar_channels=2, optical_channels=3, node_feature_dim=10, num_downs=4, use_graph_branch=False
        )
        standalone = UNetGenerator(in_channels=2, out_channels=3, num_downs=4)

        hybrid_shapes = {name: tuple(p.shape) for name, p in hybrid.unet.named_parameters()}
        standalone_shapes = {name: tuple(p.shape) for name, p in standalone.named_parameters()}
        assert hybrid_shapes == standalone_shapes

    def test_graph_branch_enabled_gives_wrapped_unet_extra_input_channels(self):
        """The other half of the same claim: *enabling* the graph branch must change the wrapped
        U-Net's first-layer input channel count (sar_channels + optical_channels), since that's
        the actual mechanism by which the structural prior reaches the decoder."""
        with_graph = GraphHybridGenerator(
            sar_channels=2, optical_channels=3, node_feature_dim=10, num_downs=4, use_graph_branch=True
        )
        first_conv = with_graph.unet.downs[0].model[0]
        assert first_conv.in_channels == 2 + 3
