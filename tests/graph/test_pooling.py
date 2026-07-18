"""Tests for src/graph/pooling.py — the pixel<->node scatter pool/unpool operations."""

import numpy as np
import pytest

from src.graph.pooling import get_node_mask, scatter_pool, unpool


def _two_region_image():
    """
    A tiny 4x4, 2-channel image split into two known regions by a label map, with known,
    hand-computable per-region means -- so scatter_pool's output can be checked against an exact
    expected value, not just "some array of the right shape."

    Top half (label 1): channel 0 = 10 everywhere, channel 1 = 100 everywhere -> mean (10, 100).
    Bottom half (label 2): channel 0 = 20 everywhere, channel 1 = 200 everywhere -> mean (20, 200).
    """
    image = np.zeros((4, 4, 2), dtype=np.float32)
    image[:2, :, 0] = 10
    image[:2, :, 1] = 100
    image[2:, :, 0] = 20
    image[2:, :, 1] = 200

    labels = np.zeros((4, 4), dtype=np.int64)
    labels[:2] = 1
    labels[2:] = 2
    return image, labels


class TestScatterPoolMean:
    def test_computes_exact_per_region_mean(self):
        image, labels = _two_region_image()

        pooled = scatter_pool(image, labels, reduction="mean")

        assert set(pooled.keys()) == {1, 2}
        np.testing.assert_allclose(pooled[1], [10, 100])
        np.testing.assert_allclose(pooled[2], [20, 200])

    def test_works_with_2_channels_not_just_3(self):
        # This is the exact case that broke skimage's rag_mean_color (hardcoded 3-channel
        # accumulator) -- a 2-channel "SAR-like" feature map must work here without special-casing.
        image, labels = _two_region_image()
        pooled = scatter_pool(image, labels)
        assert pooled[1].shape == (2,)

    def test_works_with_more_than_3_channels(self):
        # The other direction: this must also work for e.g. 12-band Sentinel-2 optical, which
        # rag_mean_color's hardcoded 3-element accumulator would equally have broken on.
        rng = np.random.default_rng(0)
        image = rng.random((4, 4, 12)).astype(np.float32)
        labels = np.zeros((4, 4), dtype=np.int64)
        labels[2:] = 1

        pooled = scatter_pool(image, labels)

        assert pooled[0].shape == (12,)
        np.testing.assert_allclose(pooled[0], image[:2].mean(axis=(0, 1)), rtol=1e-5)

    def test_raises_on_shape_mismatch(self):
        image = np.zeros((4, 4, 2))
        labels = np.zeros((5, 5), dtype=np.int64)
        with pytest.raises(ValueError, match="does not match"):
            scatter_pool(image, labels)


class TestScatterPoolMax:
    def test_computes_exact_per_region_max(self):
        image = np.zeros((4, 4, 1), dtype=np.float32)
        labels = np.zeros((4, 4), dtype=np.int64)
        image[0, 0, 0] = 5  # one high pixel in an otherwise-zero region
        image[2, 2, 0] = 9

        pooled = scatter_pool(image, labels, reduction="max")

        assert pooled[0][0] == 9

    def test_rejects_unknown_reduction(self):
        image = np.zeros((4, 4, 1))
        labels = np.zeros((4, 4), dtype=np.int64)
        with pytest.raises(ValueError, match="reduction"):
            scatter_pool(image, labels, reduction="median")


class TestUnpool:
    def test_broadcasts_node_vectors_back_to_pixels(self):
        _, labels = _two_region_image()
        node_features = {1: np.array([10.0, 100.0]), 2: np.array([20.0, 200.0])}

        broadcast = unpool(node_features, labels)

        assert broadcast.shape == (4, 4, 2)
        np.testing.assert_allclose(broadcast[0, 0], [10, 100])
        np.testing.assert_allclose(broadcast[3, 3], [20, 200])

    def test_pool_then_unpool_reconstructs_a_piecewise_constant_image(self):
        # A meaningful round-trip property: pooling a piecewise-constant image to per-node means
        # and then unpooling should reconstruct it exactly (each region was already uniform, so
        # its mean IS its value -- this is the "coarse regional prior" property the architecture
        # relies on in docs/RESEARCH_PLAN.md §6 step 6).
        image, labels = _two_region_image()

        pooled = scatter_pool(image, labels)
        reconstructed = unpool(pooled, labels)

        np.testing.assert_allclose(reconstructed, image)


class TestGetNodeMask:
    def test_returns_boolean_mask_for_the_right_region(self):
        _, labels = _two_region_image()

        mask = get_node_mask(labels, node_id=1)

        assert mask.dtype == bool
        assert mask.sum() == 8  # top half of a 4x4 grid = 8 pixels
        assert np.all(labels[mask] == 1)
