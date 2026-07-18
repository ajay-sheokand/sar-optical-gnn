#!/usr/bin/env python
"""
Render one real dataset sample as a 4-panel figure: SAR image, optical image, the SLIC superpixel
segmentation, and the Region Adjacency Graph drawn on top of it.

Why this script exists: everything built in M0-M2 (docs/BUILD_LOG.md) was verified through
numbers — test assertions, benchmark timings, "10,108 pairs found." Numbers confirm correctness
but don't make the *pipeline itself* visible: what does a real SAR patch actually look like? Does
the segmentation follow anything sensible, or is it noise? Do the graph's edges actually look like
they connect neighboring regions? This renders the same real data used throughout M0-M2 so that
can be checked by eye, not just trusted from a test suite.

Usage:
    python -m scripts.visualize_sample --dataset sarptical \
        --root data/sarptical/extracted/patch_SAR_OPT_SQUARE --index 0 --out outputs/sample_0.png

    # A random sample instead of a specific index:
    python -m scripts.visualize_sample --dataset sarptical \
        --root data/sarptical/extracted/patch_SAR_OPT_SQUARE --random --out outputs/random.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries

from src.graph_builder import build_graph_from_image

matplotlib.use("Agg")  # write straight to a file -- no display server needed/assumed


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Turn an arbitrary-channel-count (H, W, C) array into something matplotlib's imshow can render
    sensibly: grayscale for 1 channel, an RGB-ish composite for anything else (first 3 channels
    if there are >=3, or VV/VH/VV for exactly 2 -- SAR's dual-pol case, a standard SAR false-color
    trick since there's no natural third channel to show). Each channel is independently
    percentile-stretched (2nd-98th) rather than min-max normalized, since SAR backscatter in
    particular tends to have a few extreme outlier pixels that would otherwise wash out everything
    else if a true min/max were used.
    """
    num_channels = image.shape[2]

    if num_channels == 1:
        display = image[:, :, 0]
    elif num_channels == 2:
        display = np.stack([image[:, :, 0], image[:, :, 1], image[:, :, 0]], axis=-1)
    else:
        display = image[:, :, :3]

    display = display.astype(np.float32)
    low, high = np.percentile(display, [2, 98])
    if high <= low:
        high = low + 1e-6
    display = np.clip((display - low) / (high - low), 0, 1)
    return display


def _node_centroids(labels: np.ndarray) -> dict[int, tuple[float, float]]:
    """(row, col) centroid per superpixel label -- where graph nodes get drawn."""
    return {int(region.label): region.centroid for region in regionprops(labels)}


def render_sample_figure(sar: np.ndarray, optical: np.ndarray, num_segments: int = 100):
    """
    Build the 4-panel figure. Split out from main() so it's callable directly (e.g. from a
    notebook, or a test) without going through argparse/file I/O.

    Returns:
        (figure, graph, labels) -- the figure for saving/display, plus the graph and label map in
        case the caller wants to inspect them further (e.g. print node/edge counts alongside the
        image, which main() below does).
    """
    graph, labels = build_graph_from_image(sar, num_segments=num_segments)
    centroids = _node_centroids(labels)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(_normalize_for_display(sar), cmap="gray" if sar.shape[2] == 1 else None)
    axes[0, 0].set_title(f"SAR ({sar.shape[2]} channel(s))")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_normalize_for_display(optical))
    axes[0, 1].set_title(f"Optical ({optical.shape[2]} channel(s))")
    axes[0, 1].axis("off")

    sar_display = _normalize_for_display(sar)
    sar_rgb_for_boundaries = (
        np.stack([sar_display] * 3, axis=-1) if sar_display.ndim == 2 else sar_display
    )
    boundaries_image = mark_boundaries(sar_rgb_for_boundaries, labels, color=(1, 1, 0))
    axes[1, 0].imshow(boundaries_image)
    axes[1, 0].set_title(f"SLIC superpixels (requested {num_segments}, got {graph.number_of_nodes()})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(sar_rgb_for_boundaries, cmap="gray" if sar.shape[2] == 1 else None)
    for u, v in graph.edges():
        row_u, col_u = centroids[u]
        row_v, col_v = centroids[v]
        axes[1, 1].plot([col_u, col_v], [row_u, row_v], color="cyan", linewidth=0.7, alpha=0.8)
    node_rows = [c[0] for c in centroids.values()]
    node_cols = [c[1] for c in centroids.values()]
    axes[1, 1].scatter(node_cols, node_rows, s=8, color="red", zorder=3)
    axes[1, 1].set_title(f"Region Adjacency Graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
    axes[1, 1].axis("off")

    fig.tight_layout()
    return fig, graph, labels


def _load_dataset(name: str, root: str):
    """Same dispatch as scripts/build_graphs_offline.py -- kept in sync manually since both are
    small; if a third script needs this, it should move to a shared module."""
    if name == "bigearthnet":
        from src.datasets.bigearthnet import BigEarthNetSAROptical

        return BigEarthNetSAROptical(root=root)
    if name == "sen12ms":
        from src.datasets.sen12ms import SEN12MSSAROptical

        return SEN12MSSAROptical(root=root)
    if name == "sen1_2":
        from src.datasets.sen1_2 import SEN1_2Dataset

        return SEN1_2Dataset(root=root)
    if name == "sarptical":
        from src.datasets.sarptical import SARpticalDataset

        return SARpticalDataset(root=root)
    raise ValueError(f"unknown dataset {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["bigearthnet", "sen12ms", "sen1_2", "sarptical"])
    parser.add_argument("--root", required=True, help="dataset root directory (must already be downloaded)")
    parser.add_argument("--index", type=int, default=0, help="sample index to render")
    parser.add_argument("--random", action="store_true", help="pick a random index instead of --index")
    parser.add_argument("--num-segments", type=int, default=100)
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--seed", type=int, default=None, help="random seed, only used with --random")
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset, args.root)

    index = args.index
    if args.random:
        rng = np.random.default_rng(args.seed)
        index = int(rng.integers(0, len(dataset)))

    sample = dataset[index]
    print(f"Rendering {args.dataset}[{index}] (of {len(dataset)} total samples)...")

    fig, graph, labels = render_sample_figure(sample["sar"], sample["optical"], num_segments=args.num_segments)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved to {out_path}")
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


if __name__ == "__main__":
    main()
