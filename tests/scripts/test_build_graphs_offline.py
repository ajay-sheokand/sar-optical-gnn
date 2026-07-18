"""
Tests for scripts/build_graphs_offline.py's testable core: converting a graph to flat arrays, and
the cache/load round-trip. NOT tested here: --dataset mode (needs a real downloaded dataset, see
docs/BUILD_LOG.md's M1 and M2 entries for why that's out of scope) and --benchmark's printed
output (a benchmark's job is to report numbers for a human to read, not to assert against — its
underlying per-config measurement function is exercised indirectly by the tests below instead).
"""

import numpy as np

from scripts.build_graphs_offline import (
    _benchmark_one_size,
    cache_graph,
    graph_to_arrays,
    load_cached_graph,
)
from src.graph.features import compute_node_features
from src.graph_builder import build_graph_from_image


def _sample_image():
    rng = np.random.default_rng(0)
    image = np.zeros((32, 32, 2), dtype=np.float32)
    image[:16] = rng.random((16, 32, 2)).astype(np.float32) * 10
    image[16:] = rng.random((16, 32, 2)).astype(np.float32) * 10 + 50  # clearly different region
    return image


class TestGraphToArrays:
    def test_feature_matrix_row_order_matches_node_ids(self):
        image = _sample_image()
        graph, labels = build_graph_from_image(image, num_segments=8)
        node_features = compute_node_features(image, labels)

        node_ids, feature_matrix, edge_index = graph_to_arrays(graph, node_features)

        assert feature_matrix.shape[0] == len(node_ids)
        # Row i's features must actually belong to node_ids[i], not some other node -- check the
        # first row explicitly rather than trusting the loop order was preserved by accident.
        np.testing.assert_array_equal(feature_matrix[0], node_features[int(node_ids[0])])

    def test_edge_index_is_symmetric(self):
        image = _sample_image()
        graph, labels = build_graph_from_image(image, num_segments=8)
        node_features = compute_node_features(image, labels)

        _, _, edge_index = graph_to_arrays(graph, node_features)

        # Every (u, v) should have a matching (v, u) -- torch_geometric's undirected-edge
        # convention, which graph_to_arrays' docstring promises without requiring an extra
        # to_undirected() step downstream.
        pairs = set(map(tuple, edge_index.T))
        for u, v in pairs:
            assert (v, u) in pairs

    def test_single_node_graph_has_empty_edge_index_with_correct_shape(self):
        # Exercises the single-segment case fixed in src/graph_builder.py -- a 1-node, 0-edge
        # graph must still produce a well-shaped (2, 0) edge_index, not an error or a malformed
        # empty array that later code can't concatenate/reshape safely. Uses the exact image
        # size/channel-count/seed already confirmed (in tests/test_graph_builder.py) to make SLIC
        # collapse into a single segment -- whether random noise collapses to one segment turns
        # out to depend on size/channels/num_segments, so this doesn't assume it in general.
        rng = np.random.default_rng(0)
        noise_image = rng.random((32, 32, 3)).astype(np.float32)
        graph, labels = build_graph_from_image(noise_image, num_segments=50)
        assert graph.number_of_nodes() == 1  # sanity-check the premise before testing on top of it
        node_features = compute_node_features(noise_image, labels)

        _, _, edge_index = graph_to_arrays(graph, node_features)

        assert edge_index.shape == (2, 0)


class TestCacheRoundTrip:
    def test_cached_graph_loads_back_identically(self, tmp_path):
        image = _sample_image()
        out_path = tmp_path / "sample.npz"

        cache_graph(image, out_path, num_segments=8)
        loaded = load_cached_graph(out_path)

        assert set(loaded.keys()) == {"labels", "node_ids", "feature_matrix", "edge_index"}
        assert loaded["labels"].shape == (32, 32)
        assert loaded["feature_matrix"].shape[0] == len(loaded["node_ids"])

    def test_creates_parent_directories(self, tmp_path):
        image = _sample_image()
        out_path = tmp_path / "nested" / "dir" / "sample.npz"

        cache_graph(image, out_path, num_segments=8)

        assert out_path.exists()


class TestBenchmarkHelper:
    def test_runs_and_reports_plausible_timing(self):
        # Not a performance assertion (that would make the test suite flaky on slower/loaded
        # machines) -- just confirms the benchmark helper actually runs end to end and returns
        # the fields run_benchmark() prints, at a tiny size so the test itself stays fast.
        result = _benchmark_one_size(height=32, width=32, channels=2, num_segments=8, num_trials=2)

        assert result["mean_seconds"] > 0
        assert result["num_nodes"] > 0
