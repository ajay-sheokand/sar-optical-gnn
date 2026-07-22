"""
Tests for src/datasets/graph_dataset.py's GraphHybridDataset -- pairs a real cached graph
(scripts/build_graphs_offline.py's actual cache_graph(), not a synthetic stand-in) with a fake
SAR/optical base dataset, so the label-remapping and node-target-pooling logic gets checked
against a real graph's actual output shape, not an assumption about what one looks like.
"""

import numpy as np
import pytest

from scripts.build_graphs_offline import cache_graph, load_cached_graph
from src.datasets.graph_dataset import GraphHybridDataset, normalize_node_features


class TestNormalizeNodeFeatures:
    def test_output_has_roughly_zero_mean_and_unit_variance_per_column(self):
        rng = np.random.default_rng(0)
        # Deliberately wildly different per-column scales -- e.g. col 0 like `area` (hundreds),
        # col 1 like `orientation` (small radians) -- the exact real-world mismatch that caused
        # GraphBranch's tanh to saturate before this normalization existed.
        raw = np.stack([rng.uniform(50, 500, size=20), rng.uniform(-1.5, 1.5, size=20)], axis=1).astype(np.float32)

        normalized = normalize_node_features(raw)

        assert normalized.shape == raw.shape
        assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-4)

    def test_keeps_activations_in_a_range_tanh_will_not_saturate(self):
        """The actual regression this exists to prevent: raw compute_node_features-scale values
        (area in the hundreds) blew GATConv's output up past +-200, saturating tanh completely.
        Normalized features feeding the same kind of linear layer should stay in a sane range."""
        rng = np.random.default_rng(0)
        raw = np.stack([rng.uniform(50, 500, size=10), rng.uniform(-1.5, 1.5, size=10)], axis=1).astype(np.float32)

        normalized = normalize_node_features(raw)

        assert np.abs(normalized).max() < 10.0  # comfortably inside tanh's non-saturating range

    def test_does_not_produce_nan_for_a_constant_column(self):
        """A single-node graph (or a column with zero variance) would divide by zero without the
        epsilon guard -- same class of bug as src/datasets/common.py's normalize_to_tanh_range fix,
        checked directly here rather than assumed safe by analogy."""
        raw = np.full((1, 4), 100.0, dtype=np.float32)  # one node -> every column has zero std
        normalized = normalize_node_features(raw)
        assert not np.isnan(normalized).any()


class _FakeBaseDataset:
    def __init__(self, sar_images, optical_images):
        self._sar_images = sar_images
        self._optical_images = optical_images

    def __len__(self):
        return len(self._sar_images)

    def __getitem__(self, index):
        return {"sar": self._sar_images[index], "optical": self._optical_images[index]}


@pytest.fixture
def cached_graph_setup(tmp_path):
    """Builds one real cached graph from a real (smoothed, non-degenerate -- see
    src/graph_builder.py's own test suite for why smoothed noise, not flat blocks) SAR image, and
    a matching base dataset with that same SAR image paired to a real optical image."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    noise = rng.normal(size=(64, 64, 1)).astype(np.float32)
    sar_image = gaussian_filter(noise, sigma=(3, 3, 0))
    optical_image = rng.uniform(0, 255, size=(64, 64, 3)).astype(np.float32)

    cache_graph(sar_image, tmp_path / "0000000.npz", num_segments=20)

    dataset = _FakeBaseDataset(sar_images=[sar_image], optical_images=[optical_image])
    return dataset, tmp_path


class TestGraphHybridDataset:
    def test_len_matches_base_dataset(self, cached_graph_setup):
        base_dataset, cache_dir = cached_graph_setup
        dataset = GraphHybridDataset(base_dataset, cache_dir)
        assert len(dataset) == 1

    def test_returns_all_expected_keys_with_correct_shapes(self, cached_graph_setup):
        base_dataset, cache_dir = cached_graph_setup
        dataset = GraphHybridDataset(base_dataset, cache_dir)
        sample = dataset[0]

        cached = load_cached_graph(cache_dir / "0000000.npz")
        num_nodes = cached["feature_matrix"].shape[0]
        feature_dim = cached["feature_matrix"].shape[1]

        assert sample["sar"].shape == (1, 64, 64)
        assert sample["optical"].shape == (3, 64, 64)
        assert sample["node_features"].shape == (num_nodes, feature_dim)
        assert sample["edge_index"].shape[0] == 2
        assert sample["label_map"].shape == (64, 64)
        assert sample["node_targets"].shape == (num_nodes, 3)

    def test_label_map_is_a_valid_positional_index_into_node_features(self, cached_graph_setup):
        """The core correctness property unpool_torch relies on: every value in label_map must be
        a valid row index into node_features, not a raw superpixel label value."""
        base_dataset, cache_dir = cached_graph_setup
        dataset = GraphHybridDataset(base_dataset, cache_dir)
        sample = dataset[0]

        num_nodes = sample["node_features"].shape[0]
        assert sample["label_map"].min() >= 0
        assert sample["label_map"].max() < num_nodes
        # Every node must actually be reachable from the label map -- a value that doesn't appear
        # would mean unpool_torch could produce a superpixel with no source pixels, or vice versa.
        assert set(sample["label_map"].unique().tolist()) == set(range(num_nodes))

    def test_node_targets_are_the_real_optical_images_per_node_mean(self, cached_graph_setup):
        """node_targets shouldn't just have the right shape -- it should actually equal the mean
        of the real (tanh-normalized) optical image over each node's pixels."""
        base_dataset, cache_dir = cached_graph_setup
        dataset = GraphHybridDataset(base_dataset, cache_dir)
        sample = dataset[0]

        from src.datasets.common import normalize_to_tanh_range

        optical_normalized = normalize_to_tanh_range(base_dataset[0]["optical"])
        label_map = sample["label_map"].numpy()
        node_id = 0  # an arbitrary node's positional index
        expected_mean = optical_normalized[label_map == node_id].mean(axis=0)

        assert np.allclose(sample["node_targets"][node_id].numpy(), expected_mean, atol=1e-4)

    def test_output_tensors_are_in_tanh_range(self, cached_graph_setup):
        base_dataset, cache_dir = cached_graph_setup
        dataset = GraphHybridDataset(base_dataset, cache_dir)
        sample = dataset[0]

        assert sample["sar"].min() >= -1.0 and sample["sar"].max() <= 1.0
        assert sample["optical"].min() >= -1.0 and sample["optical"].max() <= 1.0
        assert sample["node_targets"].min() >= -1.0 and sample["node_targets"].max() <= 1.0
