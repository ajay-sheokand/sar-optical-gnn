"""Tests for src/models/losses.py."""

import pytest
import torch

from src.models.losses import GANLoss, cycle_consistency_loss, identity_loss


class TestGANLoss:
    def test_lsgan_zero_loss_when_predictions_match_real_target(self):
        loss_fn = GANLoss(mode="lsgan")
        predictions = torch.ones(2, 1, 8, 8)
        loss = loss_fn(predictions, target_is_real=True)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_lsgan_zero_loss_when_predictions_match_fake_target(self):
        loss_fn = GANLoss(mode="lsgan")
        predictions = torch.zeros(2, 1, 8, 8)
        loss = loss_fn(predictions, target_is_real=False)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_lsgan_penalizes_mismatch(self):
        loss_fn = GANLoss(mode="lsgan")
        predictions = torch.zeros(2, 1, 8, 8)
        loss = loss_fn(predictions, target_is_real=True)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)  # MSE(0, 1) == 1

    def test_vanilla_mode_runs_and_penalizes_mismatch(self):
        loss_fn = GANLoss(mode="vanilla")
        predictions = torch.full((2, 1, 8, 8), -10.0)  # confidently "fake" logits
        loss = loss_fn(predictions, target_is_real=True)
        assert loss.item() > 1.0

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            GANLoss(mode="not-a-real-mode")

    def test_broadcasts_to_patch_shaped_predictions(self):
        """PatchGANDiscriminator outputs a patch map, not a scalar -- the target must match it."""
        loss_fn = GANLoss()
        predictions = torch.randn(4, 1, 14, 14)
        loss = loss_fn(predictions, target_is_real=True)
        assert loss.shape == ()  # reduced to a scalar loss value


class TestCycleConsistencyLoss:
    def test_zero_for_identical_tensors(self):
        x = torch.randn(2, 3, 16, 16)
        assert cycle_consistency_loss(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_matches_manual_l1_computation(self):
        a = torch.rand(2, 3, 16, 16)
        b = torch.rand(2, 3, 16, 16)
        expected = (a - b).abs().mean()
        assert cycle_consistency_loss(a, b).item() == pytest.approx(expected.item(), abs=1e-6)


class TestIdentityLoss:
    def test_zero_for_identical_tensors(self):
        x = torch.randn(2, 3, 16, 16)
        assert identity_loss(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_matches_manual_l1_computation(self):
        a = torch.rand(2, 3, 16, 16)
        b = torch.rand(2, 3, 16, 16)
        expected = (a - b).abs().mean()
        assert identity_loss(a, b).item() == pytest.approx(expected.item(), abs=1e-6)
