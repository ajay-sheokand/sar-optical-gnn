"""Tests for src/graph/features.py — regionprops geometric features + channel mean/std."""

import numpy as np

from src.graph.features import (
    compute_channel_statistics,
    compute_geometric_features,
    compute_node_features,
)


def _two_region_labels():
    """Two equal-sized, spatially distinct 8x8-total regions, split top/bottom."""
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[:4] = 1
    labels[4:] = 2
    return labels


class TestComputeGeometricFeatures:
    def test_returns_one_entry_per_label(self):
        labels = _two_region_labels()
        features = compute_geometric_features(labels)
        assert set(features.keys()) == {1, 2}

    def test_feature_vector_has_expected_length(self):
        # 4 named properties (area, eccentricity, orientation, extent) + 2 centroid coords = 6.
        labels = _two_region_labels()
        features = compute_geometric_features(labels)
        assert features[1].shape == (6,)

    def test_area_matches_pixel_count(self):
        labels = _two_region_labels()
        features = compute_geometric_features(labels)
        # _GEOMETRIC_PROPERTIES[0] is "area" -- first element of the vector.
        assert features[1][0] == 32  # 4 rows x 8 cols
        assert features[2][0] == 32

    def test_centroids_reflect_actual_region_position(self):
        labels = _two_region_labels()
        features = compute_geometric_features(labels)
        # Centroid (row, col) is appended after the 4 named properties -- indices 4, 5.
        top_row_centroid, bottom_row_centroid = features[1][4], features[2][4]
        assert top_row_centroid < bottom_row_centroid


class TestComputeChannelStatistics:
    def test_mean_and_std_have_expected_shape(self):
        image = np.zeros((8, 8, 2), dtype=np.float32)
        labels = _two_region_labels()

        stats = compute_channel_statistics(image, labels)

        # 2 channels -> 2 means + 2 stds = 4.
        assert stats[1].shape == (4,)

    def test_constant_region_has_zero_std(self):
        image = np.full((8, 8, 1), 5.0, dtype=np.float32)
        labels = _two_region_labels()

        stats = compute_channel_statistics(image, labels)

        # layout: [mean_0, std_0] for a single channel.
        assert stats[1][0] == 5.0
        assert stats[1][1] == 0.0

    def test_std_is_nonzero_for_varying_region(self):
        rng = np.random.default_rng(0)
        image = rng.normal(loc=0, scale=3, size=(8, 8, 1)).astype(np.float32)
        labels = _two_region_labels()

        stats = compute_channel_statistics(image, labels)

        assert stats[1][1] > 0


class TestComputeNodeFeatures:
    def test_concatenates_geometric_and_channel_stats(self):
        image = np.zeros((8, 8, 2), dtype=np.float32)
        labels = _two_region_labels()

        features = compute_node_features(image, labels)

        # 6 geometric + (2 channels * 2 for mean+std) = 10.
        assert features[1].shape == (10,)

    def test_same_labels_as_geometric_features(self):
        image = np.zeros((8, 8, 1), dtype=np.float32)
        labels = _two_region_labels()

        node_features = compute_node_features(image, labels)
        geometric = compute_geometric_features(labels)

        assert set(node_features.keys()) == set(geometric.keys())
