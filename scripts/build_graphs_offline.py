#!/usr/bin/env python
"""
Precompute and cache Region-Adjacency-Graphs for a dataset, and/or benchmark how long that takes.

Why this script exists (docs/RESEARCH_PLAN.md §7, M2): SLIC segmentation + RAG construction +
regionprops feature extraction is real per-sample work — if it's re-run from scratch every epoch
during training, it could easily become the training loop's bottleneck instead of the GPU forward/
backward pass. This script (a) benchmarks that cost directly, so M3+ doesn't have to guess whether
on-the-fly graph construction is fast enough, and (b) does the actual precompute-and-cache work
once real data is available, so training can just load a cached graph instead of rebuilding it.

Two independent modes:

    python scripts/build_graphs_offline.py --benchmark
        Runs on synthetic images at the patch sizes this project's real datasets actually use
        (120x120 BigEarthNet, 256x256 SEN12MS/SEN1-2) — needs no downloaded data, so this can be
        (and was) run and reported on immediately, in M2, rather than deferred until real data
        exists. See docs/BUILD_LOG.md's M2 entry for the actual numbers this produced.

    python scripts/build_graphs_offline.py --dataset bigearthnet --root data/bigearthnet --out data/graphs/bigearthnet
        Iterates a real dataset loader (src/datasets/*) and caches one .npz per sample containing
        everything M4's GNN will need (node feature matrix, edge index, label map) without
        re-running SLIC/RAG/regionprops at training time. NOT run as part of building this script
        -- there's no downloaded dataset yet (see docs/BUILD_LOG.md's M1 entry) -- but the code
        path is real, not a stub; it exercises the exact same cache_graph()/load_cached_graph()
        functions the benchmark and tests below do.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.graph.features import compute_node_features
from src.graph_builder import build_graph_from_image


def graph_to_arrays(
    graph, node_features: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a networkx graph + per-node feature dict into the flat array format cached to disk
    and, later, loaded directly into a torch_geometric.data.Data object (M4) without needing
    networkx at load time.

    Returns:
        node_ids: (N,) int array, sorted -- row i of `feature_matrix` is node_ids[i].
        feature_matrix: (N, F) float32 array.
        edge_index: (2, E) int array of *positional* indices into node_ids (0-indexed, not raw
            label values), with both directions of each edge included -- matching
            torch_geometric's edge_index convention for undirected graphs directly, so M4 doesn't
            need an extra to_undirected() step.
    """
    node_ids = np.array(sorted(graph.nodes()), dtype=np.int64)
    id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

    feature_matrix = np.stack([node_features[node_id] for node_id in node_ids]).astype(np.float32)

    edges = []
    for u, v in graph.edges():
        edges.append((id_to_index[u], id_to_index[v]))
        edges.append((id_to_index[v], id_to_index[u]))
    edge_index = (
        np.array(edges, dtype=np.int64).T
        if edges
        else np.zeros((2, 0), dtype=np.int64)
    )

    return node_ids, feature_matrix, edge_index


def cache_graph(image_data: np.ndarray, out_path: str | Path, num_segments: int = 100) -> None:
    """Build a graph for one image and cache it to `out_path` as a single .npz file."""
    graph, labels = build_graph_from_image(image_data, num_segments=num_segments)
    node_features = compute_node_features(image_data, labels)
    node_ids, feature_matrix, edge_index = graph_to_arrays(graph, node_features)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        labels=labels,
        node_ids=node_ids,
        feature_matrix=feature_matrix,
        edge_index=edge_index,
    )


def load_cached_graph(path: str | Path) -> dict[str, np.ndarray]:
    """Load a graph cached by cache_graph(). Returns the four arrays as a plain dict."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _benchmark_one_size(height: int, width: int, channels: int, num_segments: int, num_trials: int = 5):
    rng = np.random.default_rng(0)

    times = []
    for _ in range(num_trials):
        # Random noise is the *worst case* for this benchmark, not a realistic case (see
        # src/graph_builder.py's single-segment handling) -- it's used here deliberately because
        # it forces SLIC to do real work partitioning high-frequency variation, rather than
        # potentially short-circuiting on flatter synthetic content. Real SAR/optical patches
        # have spatial structure and should be at least this fast, typically faster.
        image = rng.random((height, width, channels)).astype(np.float32)

        start = time.perf_counter()
        graph, labels = build_graph_from_image(image, num_segments=num_segments)
        compute_node_features(image, labels)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_seconds": float(np.mean(times)),
        "std_seconds": float(np.std(times)),
        "num_nodes": graph.number_of_nodes(),
    }


def run_benchmark() -> None:
    """Benchmark SLIC + RAG + node-feature construction at this project's real patch sizes."""
    configs = [
        ("BigEarthNet SAR (120x120, 2ch)", 120, 120, 2, 100),
        ("SEN12MS SAR (256x256, 2ch)", 256, 256, 2, 100),
        ("SEN12MS optical (256x256, 13ch)", 256, 256, 13, 100),
        ("SEN12MS SAR, fine granularity (256x256, 2ch, 1000 segments)", 256, 256, 2, 1000),
    ]

    print(f"{'Config':<55} {'mean (ms)':>10} {'std (ms)':>10} {'nodes':>7}")
    print("-" * 85)
    for name, height, width, channels, num_segments in configs:
        result = _benchmark_one_size(height, width, channels, num_segments)
        print(
            f"{name:<55} {result['mean_seconds'] * 1000:>10.1f} "
            f"{result['std_seconds'] * 1000:>10.1f} {result['num_nodes']:>7}"
        )


def _load_dataset(name: str, root: str):
    """Dispatch to the right loader from src/datasets/. Kept separate from main() for testing."""
    if name == "bigearthnet":
        from src.datasets.bigearthnet import BigEarthNetSAROptical

        return BigEarthNetSAROptical(root=root)
    if name == "sen12ms":
        from src.datasets.sen12ms import SEN12MSSAROptical

        return SEN12MSSAROptical(root=root)
    if name == "sen1_2":
        from src.datasets.sen1_2 import SEN1_2Dataset

        return SEN1_2Dataset(root=root)
    raise ValueError(f"unknown dataset {name!r}; expected bigearthnet, sen12ms, or sen1_2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", action="store_true", help="run the synthetic-data throughput benchmark")
    parser.add_argument("--dataset", choices=["bigearthnet", "sen12ms", "sen1_2"], help="dataset to process")
    parser.add_argument("--root", help="dataset root directory (must already be downloaded)")
    parser.add_argument("--out", help="output directory for cached .npz graph files")
    parser.add_argument("--num-segments", type=int, default=100)
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        return

    if not (args.dataset and args.root and args.out):
        parser.error("--dataset, --root, and --out are all required unless --benchmark is set")

    dataset = _load_dataset(args.dataset, args.root)
    out_dir = Path(args.out)
    print(f"Caching {len(dataset)} graphs from {args.dataset!r} to {out_dir}...")
    for index in range(len(dataset)):
        sample = dataset[index]
        cache_graph(sample["sar"], out_dir / f"{index:07d}.npz", num_segments=args.num_segments)
        if index % 1000 == 0:
            print(f"  {index}/{len(dataset)}")
    print("Done.")


if __name__ == "__main__":
    main()
