"""
Tests for scripts/train_gnn_hybrid.py -- the M4 training driver. Uses real (tiny) SLIC-segmented
graphs cached via the real cache_graph(), not synthetic stand-ins, so the whole real pipeline
(cache -> GraphHybridDataset -> GraphHybridGenerator -> losses -> checkpoint/resume) gets
exercised end to end on data small enough to run fast.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from scripts.train_gnn_hybrid import (
    ensure_graphs_cached,
    evaluate_gnn_hybrid,
    gnn_hybrid_train_step,
    infer_dims,
    prune_checkpoints,
    train_gnn_hybrid,
)
from src.datasets.graph_dataset import GraphHybridDataset
from src.eval.metrics import TranslationMetrics
from src.models.blocks import PatchGANDiscriminator
from src.models.gnn_hybrid import GraphHybridGenerator
from src.models.losses import GANLoss


class _FakeBaseDataset:
    """Real (not degenerate) SAR content per sample -- smoothed noise, same reasoning as
    src/graph_builder.py's own test suite: SLIC needs organic spatial structure to split into more
    than one region, which flat/random-per-pixel noise won't reliably do."""

    def __init__(self, n=6, height=48, width=48, sar_channels=2, optical_channels=3, seed=0):
        rng = np.random.default_rng(seed)
        self._samples = []
        for _ in range(n):
            noise = rng.normal(size=(height, width, sar_channels)).astype(np.float32)
            sar = gaussian_filter(noise, sigma=(3, 3, 0))
            optical = rng.uniform(0, 255, size=(height, width, optical_channels)).astype(np.float32)
            self._samples.append({"sar": sar, "optical": optical})

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]


def _make_args(out, **overrides):
    defaults = dict(
        out=str(out), epochs=1, lr=2e-4, num_downs=3, gnn_hidden_dim=8, gnn_layers=2,
        no_graph_branch=False, lambda_l1=100.0, lambda_node_aux=10.0, val_fraction=0.34,
        num_workers=0, seed=0, no_resume=False, keep_every_n_checkpoints=5,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def cached_dataset(tmp_path):
    base_dataset = _FakeBaseDataset(n=6)
    cache_dir = tmp_path / "graph_cache"
    ensure_graphs_cached(base_dataset, cache_dir, num_segments=8)
    return GraphHybridDataset(base_dataset, cache_dir)


class TestEnsureGraphsCached:
    def test_caches_one_file_per_sample(self, tmp_path):
        base_dataset = _FakeBaseDataset(n=4)
        cache_dir = tmp_path / "cache"

        ensure_graphs_cached(base_dataset, cache_dir, num_segments=8)

        assert sorted(p.name for p in cache_dir.glob("*.npz")) == [f"{i:07d}.npz" for i in range(4)]

    def test_is_idempotent_and_does_not_recompute_existing_files(self, tmp_path):
        base_dataset = _FakeBaseDataset(n=2)
        cache_dir = tmp_path / "cache"
        ensure_graphs_cached(base_dataset, cache_dir, num_segments=8)
        mtime_before = (cache_dir / "0000000.npz").stat().st_mtime

        ensure_graphs_cached(base_dataset, cache_dir, num_segments=8)
        mtime_after = (cache_dir / "0000000.npz").stat().st_mtime

        assert mtime_before == mtime_after


def test_infer_dims_matches_real_cached_graph(cached_dataset):
    sar_channels, optical_channels, node_feature_dim = infer_dims(cached_dataset)
    assert sar_channels == 2
    assert optical_channels == 3
    assert node_feature_dim == cached_dataset[0]["node_features"].shape[1]


class TestGnnHybridTrainStep:
    def test_produces_finite_losses_and_updates_both_networks(self, cached_dataset):
        device = torch.device("cpu")
        sar_channels, optical_channels, node_feature_dim = infer_dims(cached_dataset)
        generator = GraphHybridGenerator(
            sar_channels, optical_channels, node_feature_dim, num_downs=3, gnn_hidden_dim=8, gnn_layers=2
        ).to(device)
        discriminator = PatchGANDiscriminator(sar_channels + optical_channels).to(device)
        opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
        gan_loss = GANLoss()

        loader = torch.utils.data.DataLoader(cached_dataset, batch_size=1)
        batch = next(iter(loader))

        # Checked across *all* parameters, not just the first: a single GAT attention parameter
        # can legitimately get an exact-zero gradient for a specific small graph's topology (softmax
        # over very few neighbors can be locally invariant to the raw attention logits) without that
        # meaning training is broken -- see test_graph_branch_parameters_actually_receive_gradient
        # below for the real regression this project hit (tanh saturation from unnormalized node
        # features killing *every* graph-branch parameter's gradient, not just one).
        before = {name: param.clone() for name, param in generator.named_parameters()}
        losses = gnn_hybrid_train_step(generator, discriminator, opt_g, opt_d, gan_loss, batch, 100.0, 10.0, device)

        for value in losses.values():
            assert np.isfinite(value)
        changed = sum(1 for name, param in generator.named_parameters() if not torch.equal(before[name], param))
        assert changed > 0

    def test_graph_branch_parameters_actually_receive_gradient(self, cached_dataset):
        """Regression test for a real bug: GraphHybridDataset originally returned
        src.graph.features.compute_node_features' raw-scale output (pixel-count `area`,
        pixel-coordinate `centroid`, etc.) unnormalized. Fed into GATConv, activations reached
        +-200 after two layers, saturating GraphBranch's final tanh completely (output pinned to
        exactly +-1.0) -- and a saturated tanh has zero derivative, so *every* graph_branch
        parameter got exactly zero gradient. Fixed by src.datasets.graph_dataset.normalize_node_features
        (per-sample z-score normalization). Checks every graph_branch parameter specifically,
        not just "some parameter somewhere changed" -- that's the property that actually broke.
        """
        device = torch.device("cpu")
        sar_channels, optical_channels, node_feature_dim = infer_dims(cached_dataset)
        generator = GraphHybridGenerator(
            sar_channels, optical_channels, node_feature_dim, num_downs=3, gnn_hidden_dim=8, gnn_layers=2
        ).to(device)
        discriminator = PatchGANDiscriminator(sar_channels + optical_channels).to(device)
        opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
        gan_loss = GANLoss()

        loader = torch.utils.data.DataLoader(cached_dataset, batch_size=1)
        batch = next(iter(loader))

        before = {name: param.clone() for name, param in generator.graph_branch.named_parameters()}
        gnn_hybrid_train_step(generator, discriminator, opt_g, opt_d, gan_loss, batch, 100.0, 10.0, device)

        for name, param in generator.graph_branch.named_parameters():
            assert not torch.equal(before[name], param), f"graph_branch.{name} received zero gradient"


class TestEvaluateGnnHybrid:
    def test_returns_finite_metrics_with_graph_branch(self, cached_dataset):
        device = torch.device("cpu")
        sar_channels, optical_channels, node_feature_dim = infer_dims(cached_dataset)
        generator = GraphHybridGenerator(
            sar_channels, optical_channels, node_feature_dim, num_downs=3, use_graph_branch=True
        ).to(device)
        loader = torch.utils.data.DataLoader(cached_dataset, batch_size=1)
        metrics = TranslationMetrics(device=device)

        result = evaluate_gnn_hybrid(generator, loader, device, metrics)

        assert np.isfinite(result["psnr"])
        assert np.isfinite(result["ssim"])

    def test_returns_finite_metrics_without_graph_branch(self, cached_dataset):
        """The ablation path -- evaluate_gnn_hybrid must work even when the generator has no
        graph branch to feed graph tensors into."""
        device = torch.device("cpu")
        sar_channels, optical_channels, node_feature_dim = infer_dims(cached_dataset)
        generator = GraphHybridGenerator(
            sar_channels, optical_channels, node_feature_dim, num_downs=3, use_graph_branch=False
        ).to(device)
        loader = torch.utils.data.DataLoader(cached_dataset, batch_size=1)
        metrics = TranslationMetrics(device=device)

        result = evaluate_gnn_hybrid(generator, loader, device, metrics)

        assert np.isfinite(result["psnr"])
        assert np.isfinite(result["ssim"])


class TestPruneCheckpoints:
    """Regression coverage for the disk-usage fix this project needed to run M4 on Kaggle: each
    checkpoint is ~550MB (generator + discriminator + both Adam optimizer states), and an unpruned
    80-epoch run would blow past Kaggle's 20GB /kaggle/working output cap around epoch 36."""

    def test_keeps_every_nth_epoch_and_deletes_the_rest(self, tmp_path):
        for epoch in range(1, 11):
            (tmp_path / f"epoch_{epoch:04d}.pt").write_bytes(b"x")

        prune_checkpoints(tmp_path, keep_every=5, current_epoch=10)

        remaining = sorted(p.name for p in tmp_path.glob("epoch_*.pt"))
        assert remaining == ["epoch_0005.pt", "epoch_0010.pt"]

    def test_always_keeps_the_current_epoch_even_if_not_a_multiple(self, tmp_path):
        for epoch in range(1, 8):
            (tmp_path / f"epoch_{epoch:04d}.pt").write_bytes(b"x")

        prune_checkpoints(tmp_path, keep_every=5, current_epoch=7)

        remaining = sorted(p.name for p in tmp_path.glob("epoch_*.pt"))
        assert remaining == ["epoch_0005.pt", "epoch_0007.pt"]

    def test_disabled_by_zero_via_the_training_loop(self, tmp_path, cached_dataset):
        """keep_every_n_checkpoints=0 must skip pruning entirely (used by the local/smoke-test
        path, where nothing needs to disappear)."""
        device = torch.device("cpu")
        args = _make_args(tmp_path, epochs=3, keep_every_n_checkpoints=0)

        train_gnn_hybrid(cached_dataset, args, device)

        assert sorted(p.name for p in tmp_path.glob("epoch_*.pt")) == [
            "epoch_0001.pt", "epoch_0002.pt", "epoch_0003.pt",
        ]


class TestTrainGnnHybridEndToEnd:
    def test_runs_and_produces_checkpoints_and_metrics(self, tmp_path, cached_dataset):
        device = torch.device("cpu")
        args = _make_args(tmp_path, epochs=1)

        train_gnn_hybrid(cached_dataset, args, device)

        assert (tmp_path / "epoch_0001.pt").exists()
        lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["epoch"] == 1

    def test_resumes_instead_of_restarting(self, tmp_path, cached_dataset):
        device = torch.device("cpu")
        train_gnn_hybrid(cached_dataset, _make_args(tmp_path, epochs=1), device)
        train_gnn_hybrid(cached_dataset, _make_args(tmp_path, epochs=2), device)

        epochs_logged = [
            json.loads(l)["epoch"] for l in (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        ]
        assert epochs_logged == [1, 2]

    def test_no_graph_branch_ablation_runs_end_to_end(self, tmp_path, cached_dataset):
        device = torch.device("cpu")
        args = _make_args(tmp_path, epochs=1, no_graph_branch=True)

        train_gnn_hybrid(cached_dataset, args, device)

        assert (tmp_path / "epoch_0001.pt").exists()
        record = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
        assert record["train_loss"]["loss_node_aux"] == 0.0
