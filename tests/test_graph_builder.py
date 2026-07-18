"""
Tests for src/graph_builder.py — the superpixel + Region Adjacency Graph (RAG) step.

Why this file matters beyond "does it crash": this is the piece of the pipeline that turns a raw
image into the graph a GNN can operate on (see docs/UNDERSTANDING_THE_PROJECT.md §3.1 for the
regionalization/spatial-weights-matrix analogy). Bugs here are easy to miss visually — a graph
with the wrong number of nodes, or edges connecting non-adjacent regions, still "looks fine" if
you just eyeball a plot — so this needs real assertions, not just "it ran without an exception."

We use skimage's bundled astronaut() photo as a stand-in for a real, structured image (it ships
with scikit-image, no network access needed). Pure synthetic block images are used where we need
to know the *exact* number of regions in advance, which a natural photo can't guarantee (SLIC's
region count is approximate, not exact).
"""

import numpy as np
import pytest
from skimage.data import astronaut

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

    def test_single_segment_image_raises_keyerror(self):
        """
        Known, currently-unfixed limitation — documented here rather than silently tolerated.

        If SLIC collapses an image into a single segment (this reliably happens on pure random
        noise, where there's no spatial color structure for SLIC's k-means step to latch onto),
        skimage's RAG constructor ends up with *zero* graph nodes, even though the label array is
        non-empty. `rag_mean_color` then crashes with a KeyError while trying to accumulate pixel
        statistics into a node that was never created.

        This isn't something this project's SAR imagery is expected to trigger in practice (real
        remote-sensing scenes have spatial structure), but it's a real upstream skimage behavior,
        not a hypothetical. Tracked for a real fix in M2 (see docs/RESEARCH_PLAN.md §7) — when
        that's fixed, this test should be updated to assert the new, non-crashing behavior instead
        of `pytest.raises`.
        """
        rng = np.random.default_rng(0)
        noise_image = rng.random((32, 32, 3)).astype(np.float32)

        with pytest.raises(KeyError):
            build_graph_from_image(noise_image, num_segments=50)
