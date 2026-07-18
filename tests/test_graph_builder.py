"""
Tests for src/graph_builder.py — the superpixel + Region Adjacency Graph (RAG) step.

Why this file matters beyond "does it crash": this is the piece of the pipeline that turns a raw
image into the graph a GNN can operate on (see docs/UNDERSTANDING_THE_PROJECT.md §3.1 for the
regionalization/spatial-weights-matrix analogy). Bugs here are easy to miss numerically — a
"correct" node/edge count doesn't tell you whether the segmentation is actually following image
content or degenerately grid-like. That specific failure mode (see
test_segmentation_is_not_a_rigid_grid below) was only caught by actually rendering
build_graph_from_image's output against real data with scripts/visualize_sample.py, not by any
number-only test — worth remembering as a general lesson: shape/count assertions and looking at
the actual output catch different classes of bugs, neither substitutes for the other.

We use skimage's bundled astronaut() photo as a stand-in for a real, structured image (it ships
with scikit-image, no network access needed). Pure synthetic block images are used where we need
to know the *exact* number of regions in advance, which a natural photo can't guarantee (SLIC's
region count is approximate, not exact).
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from skimage.data import astronaut
from skimage.measure import regionprops

from src.graph_builder import build_graph_from_image


def make_block_image(block_size=16, grid=4):
    """
    Build a synthetic image made of a `grid` x `grid` checkerboard of flat-colored blocks.

    Why: SLIC's superpixel count is approximate (it aims for `num_segments`, doesn't guarantee it
    exactly), so for tests that need to reason about a *known* region count we hand SLIC an image
    with unambiguous, high-contrast region boundaries instead of natural photo texture. Each block
    gets a distinct random-ish color so adjacent blocks are never accidentally merged into one
    superpixel by SLIC's color-similarity criterion.
    """
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(grid, grid, 3), dtype=np.uint8)
    image = np.zeros((grid * block_size, grid * block_size, 3), dtype=np.uint8)
    for i in range(grid):
        for j in range(grid):
            image[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ] = colors[i, j]
    return image


class TestBuildGraphFromImage:
    def test_returns_graph_and_labels(self):
        """Basic contract: a networkx Graph plus a label array shaped like the image's spatial dims."""
        image = astronaut()
        graph, labels = build_graph_from_image(image, num_segments=100)

        assert labels.shape == image.shape[:2]
        assert graph.number_of_nodes() > 0
        # Every label that appears in the label map must exist as a node in the RAG.
        assert set(np.unique(labels)) == set(graph.nodes())

    def test_graph_nodes_match_unique_labels(self):
        """
        Each RAG node should correspond to exactly one superpixel label — not more, not fewer.
        This is the property that makes the RAG a legitimate stand-in for a spatial weights
        matrix (docs/UNDERSTANDING_THE_PROJECT.md §3.1): one row/column per areal unit.
        """
        image = make_block_image(block_size=16, grid=4)
        graph, labels = build_graph_from_image(image, num_segments=16)

        unique_labels = set(np.unique(labels))
        assert set(graph.nodes()) == unique_labels

    def test_more_requested_segments_gives_more_regions(self):
        """
        SLIC's `num_segments` is a target, not an exact count, but requesting more segments on a
        richly-varied image should reliably produce more actual regions — not fewer. If this
        stops holding, something is wrong with how `num_segments` is being passed through to SLIC
        (this is exactly the kind of bug that broke this project's first draft: the parameter was
        silently accepted under the wrong keyword and had no effect at all).
        """
        image = astronaut()

        _, labels_coarse = build_graph_from_image(image, num_segments=20)
        _, labels_fine = build_graph_from_image(image, num_segments=200)

        assert len(np.unique(labels_fine)) > len(np.unique(labels_coarse))

    def test_edges_only_connect_spatially_adjacent_regions(self):
        """
        A RAG edge should only exist between superpixels that actually touch in the image. We
        check this indirectly: in our 4x4 block-grid image, the block-adjacency structure is
        known by construction, so no RAG node should exceed the maximum possible neighbor count.

        `rag_mean_color` defaults to `connectivity=2` (skimage.graph._rag.rag_mean_color's
        default), which means 8-connected (Moore neighborhood) adjacency — diagonal neighbors
        count as touching, not just up/down/left/right. So the cap here is 8, not 4. (This was
        confirmed empirically: an earlier version of this test wrongly assumed 4-connectivity and
        failed on interior blocks that legitimately have diagonal neighbors — a good reminder that
        the *default* adjacency rule is worth checking explicitly rather than assumed.)
        """
        image = make_block_image(block_size=16, grid=4)
        graph, labels = build_graph_from_image(image, num_segments=16)

        max_possible_neighbors = 8
        for node in graph.nodes():
            assert graph.degree(node) <= max_possible_neighbors, (
                f"node {node} has degree {graph.degree(node)}, "
                f"exceeding the 8-neighbor (Moore) cap implied by a block-grid layout"
            )

    def test_default_num_segments_is_100(self):
        """Locks in the documented default so a future refactor can't silently change it."""
        image = astronaut()
        _, labels_default = build_graph_from_image(image)
        _, labels_explicit = build_graph_from_image(image, num_segments=100)

        assert len(np.unique(labels_default)) == len(np.unique(labels_explicit))

    def test_single_segment_image_produces_one_node_graph(self):
        """
        Regression test for a bug fixed in M2 — previously documented here (M0) as a known,
        unfixed limitation raising KeyError, now fixed. See src/graph_builder.py's module
        docstring and docs/BUILD_LOG.md's M2 entry for the full story: skimage's RAG class only
        creates a node when it detects an edge between two differently-labeled pixels, so a
        single-segment image (SLIC reliably collapses pure random noise into one segment — no
        spatial color structure for its k-means step to latch onto) used to produce a zero-node
        graph and crash downstream. It now produces a valid one-node, zero-edge graph instead.
        """
        rng = np.random.default_rng(0)
        noise_image = rng.random((32, 32, 3)).astype(np.float32)

        graph, labels = build_graph_from_image(noise_image, num_segments=50)

        assert len(np.unique(labels)) == 1
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0
        only_node = next(iter(graph.nodes()))
        assert graph.nodes[only_node]["pixel count"] == 32 * 32

    def test_works_on_2_channel_sar_like_image(self):
        """
        The multi-channel bug this project actually found in M2 (see src/graph_builder.py's
        module docstring): skimage.graph.rag_mean_color hardcodes a 3-element color accumulator
        and breaks on anything that isn't exactly 3 channels. Real Sentinel-1 SAR input is 2
        channels (VV, VH) -- this must work end to end, not just in isolation
        (tests/graph/test_pooling.py already checks the underlying scatter_pool function directly;
        this checks the full build_graph_from_image pipeline on top of it).
        """
        image = make_block_image(block_size=16, grid=4)
        sar_like = image[:, :, :2].astype(np.float32)  # first 2 of the 3 synthetic channels

        graph, labels = build_graph_from_image(sar_like, num_segments=16)

        assert graph.number_of_nodes() > 0
        only_node = next(iter(graph.nodes()))
        assert graph.nodes[only_node]["mean color"].shape == (2,)

    def test_works_on_12_channel_optical_like_image(self):
        """The other direction: full multispectral Sentinel-2 (12 bands), not just SAR's 2."""
        rng = np.random.default_rng(0)
        multispectral = rng.random((32, 32, 12)).astype(np.float32)

        graph, labels = build_graph_from_image(multispectral, num_segments=20)

        assert graph.number_of_nodes() > 0
        only_node = next(iter(graph.nodes()))
        assert graph.nodes[only_node]["mean color"].shape == (12,)

    def test_edge_weight_is_distance_between_mean_colors(self):
        """
        Locks in the specific edge-weight formula (matches skimage.graph.rag_mean_color's default
        "distance" mode: Euclidean distance between the two nodes' mean colors) against a
        hand-computable case: two large, constant-valued, spatially separated blocks (small
        images -- e.g. 4x4 -- turned out to be too small for SLIC to reliably split at all; it
        collapsed one such attempt into a single segment, the same collapse behavior exercised
        deliberately in test_single_segment_image_produces_one_node_graph above, so this test
        uses the same block size proven to produce distinct regions in make_block_image).
        """
        image = np.zeros((32, 16, 1), dtype=np.float32)
        image[:16] = 10.0
        image[16:] = 13.0  # constant, known offset -> exact expected edge weight of 3.0

        graph, labels = build_graph_from_image(image, num_segments=2)

        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        (u, v, data) = next(iter(graph.edges(data=True)))
        assert data["weight"] == pytest.approx(3.0)

    def test_segmentation_is_not_a_rigid_grid(self):
        """
        Regression test for a real finding from actually visualizing this function's output
        against downloaded SAR data (scripts/visualize_sample.py; see src/graph_builder.py's
        module docstring and docs/BUILD_LOG.md for the full story): the default `compactness`
        used to produce a near-perfectly regular grid on real SAR patches -- 100% of superpixels
        were exact rectangles -- instead of content-adaptive regions, and no purely numeric test
        (node count, edge count, shapes) caught this, because a rigid grid is a "valid" RAG by
        every one of those measures.

        The metric used here, `extent` (skimage.measure.regionprops: the fraction of a region's
        bounding box that the region itself actually fills), is what caught it: exactly 1.0 for a
        rectangle, meaningfully lower for an irregular shape. This test uses smoothed random
        noise (not real downloaded data, so it runs in any environment) specifically because it
        has organic, blob-like structure similar in spirit to real speckle texture, unlike the
        flat-colored block images used elsewhere in this file, which are *supposed* to segment
        into clean rectangles and would wrongly fail this exact check.
        """
        rng = np.random.default_rng(0)
        noise = rng.normal(size=(112, 112, 1)).astype(np.float32)
        smoothly_varying_image = gaussian_filter(noise, sigma=(3, 3, 0))

        _, labels = build_graph_from_image(smoothly_varying_image, num_segments=100)

        extents = [region.extent for region in regionprops(labels)]
        mean_extent = float(np.mean(extents))
        assert mean_extent < 0.9, (
            f"mean superpixel extent {mean_extent:.3f} is suspiciously close to 1.0 (a perfect "
            f"rectangle) -- segmentation may have degenerated into a rigid grid instead of "
            f"following image content, the exact failure mode this test guards against"
        )
