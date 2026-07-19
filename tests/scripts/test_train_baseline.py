"""
Tests for scripts/train_baseline.py's training-loop machinery. Runs real (tiny) forward/backward
passes on synthetic data rather than mocking torch out -- the risk in a training script isn't "does
it import," it's "do gradients actually flow and do losses stay finite across the pix2pix/CycleGAN
step functions," which only a real step can catch.
"""

import json

import numpy as np
import pytest
import torch

from scripts.train_baseline import (
    build_cyclegan,
    build_pix2pix,
    cyclegan_train_step,
    evaluate_cyclegan,
    evaluate_pix2pix,
    infer_channel_counts,
    log_metrics,
    pix2pix_train_step,
    save_checkpoint,
    split_dataset,
)
from src.datasets.adapter import TranslationDataset
from src.models.losses import GANLoss


class _FakeBaseDataset:
    def __init__(self, n=8, sar_shape=(64, 64, 2), optical_shape=(64, 64, 3)):
        rng = np.random.default_rng(0)
        self._samples = [
            {
                "sar": rng.normal(loc=100, scale=20, size=sar_shape).astype(np.float32),
                "optical": rng.uniform(0, 255, size=optical_shape).astype(np.float32),
            }
            for _ in range(n)
        ]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]


@pytest.fixture
def tiny_dataset():
    return TranslationDataset(_FakeBaseDataset(n=8))


class TestSplitDataset:
    def test_splits_by_val_fraction(self, tiny_dataset):
        train_set, val_set = split_dataset(tiny_dataset, val_fraction=0.25, seed=0)
        assert len(train_set) == 6
        assert len(val_set) == 2

    def test_val_set_is_never_empty_even_for_tiny_fraction(self, tiny_dataset):
        _, val_set = split_dataset(tiny_dataset, val_fraction=0.01, seed=0)
        assert len(val_set) >= 1

    def test_rejects_a_dataset_too_small_to_split(self):
        dataset = TranslationDataset(_FakeBaseDataset(n=1))
        with pytest.raises(ValueError, match="too small"):
            split_dataset(dataset, val_fraction=0.5, seed=0)


def test_infer_channel_counts(tiny_dataset):
    sar_channels, optical_channels = infer_channel_counts(tiny_dataset)
    assert sar_channels == 2
    assert optical_channels == 3


class TestPix2PixStep:
    def test_produces_finite_losses_and_updates_generator_weights(self, tiny_dataset):
        device = torch.device("cpu")
        generator, discriminator = build_pix2pix(2, 3, num_downs=3, device=device)
        opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
        gan_loss = GANLoss()

        before = next(generator.parameters()).clone()
        batch = {"sar": torch.stack([tiny_dataset[i]["sar"] for i in range(2)]),
                  "optical": torch.stack([tiny_dataset[i]["optical"] for i in range(2)])}

        losses = pix2pix_train_step(
            generator, discriminator, opt_g, opt_d, gan_loss, batch["sar"], batch["optical"], lambda_l1=100.0, device=device
        )

        for value in losses.values():
            assert np.isfinite(value)
        after = next(generator.parameters())
        assert not torch.equal(before, after)

    def test_evaluate_returns_finite_psnr_and_ssim(self, tiny_dataset):
        device = torch.device("cpu")
        generator, _ = build_pix2pix(2, 3, num_downs=3, device=device)
        loader = torch.utils.data.DataLoader(tiny_dataset, batch_size=4)

        result = evaluate_pix2pix(generator, loader, device)

        assert np.isfinite(result["psnr"])
        assert np.isfinite(result["ssim"])
        assert "fid" in result  # optical has 3 channels in this fixture, so FID should run


class TestCycleGANStep:
    def test_produces_finite_losses_and_updates_both_generators(self, tiny_dataset):
        device = torch.device("cpu")
        g_ab, g_ba, d_a, d_b = build_cyclegan(2, 3, n_residual_blocks=1, device=device)
        opt_g = torch.optim.Adam(list(g_ab.parameters()) + list(g_ba.parameters()), lr=2e-4)
        opt_d = torch.optim.Adam(list(d_a.parameters()) + list(d_b.parameters()), lr=2e-4)
        gan_loss = GANLoss()

        before_ab = next(g_ab.parameters()).clone()
        before_ba = next(g_ba.parameters()).clone()
        batch = {"sar": torch.stack([tiny_dataset[i]["sar"] for i in range(2)]),
                  "optical": torch.stack([tiny_dataset[i]["optical"] for i in range(2)])}

        losses = cyclegan_train_step(
            g_ab, g_ba, d_a, d_b, opt_g, opt_d, gan_loss,
            batch["sar"], batch["optical"], lambda_cycle=10.0, lambda_identity=0.0, device=device,
        )

        for value in losses.values():
            assert np.isfinite(value)
        assert not torch.equal(before_ab, next(g_ab.parameters()))
        assert not torch.equal(before_ba, next(g_ba.parameters()))
        assert losses["loss_identity"] == 0.0  # disabled via lambda_identity=0.0

    def test_identity_loss_is_skipped_when_channels_mismatch_even_if_requested(self, tiny_dataset):
        """SAR (2ch) and optical (3ch) can't type-check identity_loss (see cyclegan.py's
        docstring) -- requesting it anyway (lambda_identity > 0) must not crash, and must not
        silently pretend to apply it either (checked via the returned loss value staying at 0)."""
        device = torch.device("cpu")
        g_ab, g_ba, d_a, d_b = build_cyclegan(2, 3, n_residual_blocks=1, device=device)
        opt_g = torch.optim.Adam(list(g_ab.parameters()) + list(g_ba.parameters()), lr=2e-4)
        opt_d = torch.optim.Adam(list(d_a.parameters()) + list(d_b.parameters()), lr=2e-4)
        gan_loss = GANLoss()
        batch = {"sar": torch.stack([tiny_dataset[i]["sar"] for i in range(2)]),
                  "optical": torch.stack([tiny_dataset[i]["optical"] for i in range(2)])}

        losses = cyclegan_train_step(
            g_ab, g_ba, d_a, d_b, opt_g, opt_d, gan_loss,
            batch["sar"], batch["optical"], lambda_cycle=10.0, lambda_identity=5.0, device=device,
        )

        assert losses["loss_identity"] == 0.0

    def test_identity_loss_applies_when_channels_match(self):
        """Same-channel-count case (e.g. an ablation with SAR replicated to 3 channels) --
        identity loss should actually compute a nonzero value here."""
        dataset = TranslationDataset(_FakeBaseDataset(n=4, sar_shape=(64, 64, 3), optical_shape=(64, 64, 3)))
        device = torch.device("cpu")
        g_ab, g_ba, d_a, d_b = build_cyclegan(3, 3, n_residual_blocks=1, device=device)
        opt_g = torch.optim.Adam(list(g_ab.parameters()) + list(g_ba.parameters()), lr=2e-4)
        opt_d = torch.optim.Adam(list(d_a.parameters()) + list(d_b.parameters()), lr=2e-4)
        gan_loss = GANLoss()
        batch = {"sar": torch.stack([dataset[i]["sar"] for i in range(2)]),
                  "optical": torch.stack([dataset[i]["optical"] for i in range(2)])}

        losses = cyclegan_train_step(
            g_ab, g_ba, d_a, d_b, opt_g, opt_d, gan_loss,
            batch["sar"], batch["optical"], lambda_cycle=10.0, lambda_identity=5.0, device=device,
        )

        assert losses["loss_identity"] > 0.0

    def test_evaluate_cyclegan_returns_finite_metrics(self, tiny_dataset):
        device = torch.device("cpu")
        g_ab, _, _, _ = build_cyclegan(2, 3, n_residual_blocks=1, device=device)
        loader = torch.utils.data.DataLoader(tiny_dataset, batch_size=4)

        result = evaluate_cyclegan(g_ab, loader, device)

        assert np.isfinite(result["psnr"])
        assert np.isfinite(result["ssim"])


class TestCheckpointingAndLogging:
    def test_save_checkpoint_creates_a_loadable_file(self, tmp_path):
        device = torch.device("cpu")
        generator, discriminator = build_pix2pix(2, 3, num_downs=3, device=device)
        state = {"generator": generator.state_dict(), "discriminator": discriminator.state_dict()}

        path = save_checkpoint(state, tmp_path, epoch=1)

        assert path.exists()
        loaded = torch.load(path, weights_only=True)
        assert set(loaded["generator"].keys()) == set(generator.state_dict().keys())

    def test_log_metrics_appends_jsonl_lines(self, tmp_path):
        log_metrics(tmp_path, {"epoch": 1, "value": 1.0})
        log_metrics(tmp_path, {"epoch": 2, "value": 2.0})

        lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["epoch"] == 1
        assert json.loads(lines[1])["epoch"] == 2
