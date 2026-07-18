"""
Superpixel segmentation + Region Adjacency Graph (RAG) construction — turns a raw image into the
graph a GNN can operate on. See docs/UNDERSTANDING_THE_PROJECT.md §3.1 for the regionalization /
spatial-weights-matrix analogy behind what this file is doing conceptually.

This module builds the RAG itself rather than using skimage.graph.rag_mean_color, after two real
bugs were found in that function during M2 while testing it against multi-channel SAR-like input
(see docs/BUILD_LOG.md's M2 entry for the full story):

1. rag_mean_color hardcodes `'total color': np.array([0, 0, 0], dtype=np.float64)` — a fixed
   3-element accumulator, regardless of how many channels the input image actually has. It works
   by accident for 3-channel RGB and silently breaks (a shape-mismatch ValueError) for anything
   else — including 2-channel SAR (VV/VH) and 12-channel Sentinel-2 optical, both of which this
   project needs to feed through here eventually.
2. skimage.graph.RAG only creates a node for a label when it detects an *edge* (two adjacent
   pixels with different labels). A single-segment image — reliably produced by SLIC on pure
   noise, with no spatial color structure to latch onto — has zero such edges, so it ends up with
   a *zero-node* graph even though its label array isn't empty. (This was tracked as a documented,
   known-but-unfixed limitation from M0 onward; it's fixed here as of M2.)

The fix for both: use skimage's RAG class only for what it does correctly (detecting adjacency
structure), and compute per-node mean color ourselves via src.graph.pooling.scatter_pool, which is
channel-count-agnostic and vectorized (no per-pixel Python loop). Any node missing from the RAG
after adjacency detection (bug 2) gets added explicitly.
"""

import networkx as nx
import numpy as np
from skimage.graph import RAG
from skimage.segmentation import slic

from src.graph.pooling import scatter_pool


def build_graph_from_image(image_data, num_segments=100, channel_axis=-1):
    """
    Segment `image_data` into superpixels and build the Region Adjacency Graph connecting them.

    Args:
        image_data: (H, W, C) array. C can be any number of channels — 1 (grayscale/single-pol
            SAR), 2 (dual-pol SAR, VV+VH), 3 (RGB), 12+ (multispectral optical) all work.
        num_segments: target superpixel count passed to SLIC (approximate, not exact — see
            tests/test_graph_builder.py's test_more_requested_segments_gives_more_regions for why
            this is a target, not a guarantee).
        channel_axis: passed straight through to `slic()`. The default (-1, "last axis is
            channel") already matches this project's (H, W, C) convention and this project's
            installed skimage version already defaults to the same value — this parameter exists
            for explicitness and to protect against a future skimage version changing its default
            silently, not because a real bug was found here (unlike the two described above).

    Returns:
        (nx_graph, labels) — nx_graph is a networkx.Graph with one node per superpixel (node id =
        label value), each with a 'mean color' attribute ((C,) array) and 'pixel count'; edges
        have a 'weight' attribute (Euclidean distance between the two nodes' mean colors, matching
        skimage.graph.rag_mean_color's default "distance" mode). labels is the (H, W) integer
        label map SLIC produced.
    """
    labels = slic(
        image_data,
        n_segments=num_segments,
        compactness=10,
        start_label=1,
        channel_axis=channel_axis,
    )

    rag = _build_rag(image_data, labels)
    return nx.Graph(rag), labels


def _build_rag(image_data, labels, connectivity=2):
    """
    Build a Region Adjacency Graph with per-node mean-color features, for images with any number
    of channels. See this module's docstring for why this doesn't just call
    skimage.graph.rag_mean_color.
    """
    rag = RAG(labels, connectivity=connectivity)

    # skimage's RAG only adds a node when it detects an edge to a differently-labeled neighbor
    # (bug 2 in the module docstring) -- make sure every label that actually appears in the
    # segmentation has a node, even ones with no detected neighbors (e.g. the whole image is one
    # segment, or a segment is fully enclosed by mask/background and never gets visited by the
    # edge-detection pass).
    for label in np.unique(labels):
        if label not in rag:
            rag.add_node(label)

    mean_colors = scatter_pool(image_data, labels, reduction="mean")
    for node, color in mean_colors.items():
        rag.nodes[node]["mean color"] = color
        rag.nodes[node]["pixel count"] = int(np.sum(labels == node))

    for u, v, data in rag.edges(data=True):
        diff = rag.nodes[u]["mean color"] - rag.nodes[v]["mean color"]
        data["weight"] = float(np.linalg.norm(diff))

    return rag
