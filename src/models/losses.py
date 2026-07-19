"""
Loss functions for the M3 baselines.

GANLoss covers both pix2pix and CycleGAN's adversarial term. The cycle-consistency and identity
losses below are CycleGAN-specific (Zhu et al. 2017) and are plain L1 -- kept here rather than
inlined in the training script since they're exact formulas from the paper worth naming and
testing directly, not incidental training-loop glue.
"""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    Adversarial loss against a constant real/fake target, broadcast to match whatever spatial
    shape the PatchGAN discriminator outputs (its output is a patch map, not a single scalar --
    see src/models/blocks.py's PatchGANDiscriminator docstring for why).

    mode="lsgan" (default) uses MSE against target labels 1.0/0.0, matching CycleGAN's paper
    (found to train more stably than the original pix2pix paper's vanilla sigmoid cross-entropy,
    which mode="vanilla" reproduces if needed for a direct pix2pix-paper comparison).
    """

    def __init__(self, mode="lsgan"):
        super().__init__()
        if mode not in ("lsgan", "vanilla"):
            raise ValueError(f"mode must be 'lsgan' or 'vanilla', got {mode!r}")
        self.mode = mode
        self.loss_fn = nn.MSELoss() if mode == "lsgan" else nn.BCEWithLogitsLoss()

    def forward(self, predictions, target_is_real):
        target = torch.ones_like(predictions) if target_is_real else torch.zeros_like(predictions)
        return self.loss_fn(predictions, target)


def cycle_consistency_loss(reconstructed, original):
    """||F(G(x)) - x||_1 -- the loss that lets CycleGAN train without paired ground truth."""
    return nn.functional.l1_loss(reconstructed, original)


def identity_loss(generated_identity, original):
    """||G(y) - y||_1 for y already in G's target domain -- regularizes color/tone preservation,
    per the CycleGAN paper's optional identity-mapping term."""
    return nn.functional.l1_loss(generated_identity, original)
