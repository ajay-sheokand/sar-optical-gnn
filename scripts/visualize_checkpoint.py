#!/usr/bin/env python
"""
Render what a training-in-progress checkpoint is actually producing: SAR input, this checkpoint's
generated optical, and the real optical ground truth, side by side, on one real dataset sample.

Why this script exists: a running training job's log lines (loss values, PSNR/SSIM numbers) prove
the loop is executing, but not what it's actually learning to produce -- the same lesson M2 already
learned the hard way (docs/BUILD_LOG.md: a "valid" numeric result hid a real segmentation bug that
only rendering the actual output caught). This is that same check applied to a live training run:
load whatever checkpoint currently exists on disk and look at its real output, not just its logged
metrics.

Usage:
    python -m scripts.visualize_checkpoint --model pix2pix \
        --checkpoint outputs/sen1_2_pix2pix/epoch_0001.pt \
        --dataset sen1_2 --root data/sen1_2 --random --out outputs/checkpoint_check.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.visualize_sample import _load_dataset, _normalize_for_display
from src.datasets.common import hwc_to_chw_tensor, normalize_to_tanh_range
from src.models.cyclegan import ResnetGenerator
from src.models.pix2pix import UNetGenerator


def load_generator(
    checkpoint_path, model_type, sar_channels, optical_channels, device, num_downs=7, n_residual_blocks=9
):
    """Build the right generator architecture and load its trained weights from a
    scripts/train_baseline.py checkpoint. pix2pix checkpoints store the generator under the key
    "generator"; CycleGAN checkpoints store two (SAR->optical and optical->SAR) -- "g_ab" is the
    SAR->optical direction, the one comparable to pix2pix's single generator."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if model_type == "pix2pix":
        generator = UNetGenerator(sar_channels, optical_channels, num_downs=num_downs)
        generator.load_state_dict(state["generator"])
    else:
        generator = ResnetGenerator(sar_channels, optical_channels, n_residual_blocks=n_residual_blocks)
        generator.load_state_dict(state["g_ab"])

    generator.to(device).eval()
    return generator


@torch.no_grad()
def run_generator(generator, sar_raw, device):
    """sar_raw: (H, W, C) raw-range numpy array. Returns (H, W, C) tanh-range numpy output."""
    sar_tensor = hwc_to_chw_tensor(normalize_to_tanh_range(sar_raw)).unsqueeze(0).to(device)
    fake_tensor = generator(sar_tensor)
    return fake_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)


def render_checkpoint_figure(sar_raw, optical_raw, generator, device, title_suffix=""):
    """3-panel figure: SAR input, this checkpoint's generated optical, real optical ground truth.
    Split out from main() so it's callable directly (e.g. to check on a run mid-training without
    going through argparse), matching scripts/visualize_sample.py's render_sample_figure pattern."""
    fake = run_generator(generator, sar_raw, device)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(_normalize_for_display(sar_raw), cmap="gray" if sar_raw.shape[2] == 1 else None)
    axes[0].set_title(f"SAR (input, {sar_raw.shape[2]} channel(s))")
    axes[0].axis("off")

    axes[1].imshow(_normalize_for_display(fake))
    axes[1].set_title(f"Generated optical{title_suffix}")
    axes[1].axis("off")

    axes[2].imshow(_normalize_for_display(optical_raw))
    axes[2].set_title("Real optical (ground truth)")
    axes[2].axis("off")

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["pix2pix", "cyclegan"], required=True)
    parser.add_argument("--checkpoint", required=True, help="path to a scripts/train_baseline.py .pt checkpoint")
    parser.add_argument("--dataset", required=True, choices=["bigearthnet", "sen12ms", "sen1_2", "sarptical"])
    parser.add_argument("--root", required=True, help="dataset root directory (must already be downloaded)")
    parser.add_argument("--index", type=int, default=0, help="sample index to render")
    parser.add_argument("--random", action="store_true", help="pick a random index instead of --index")
    parser.add_argument("--seed", type=int, default=None, help="random seed, only used with --random")
    parser.add_argument("--num-downs", type=int, default=7, help="pix2pix U-Net depth (must match training)")
    parser.add_argument("--n-residual-blocks", type=int, default=9, help="CycleGAN ResNet depth (must match training)")
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset, args.root)
    index = args.index
    if args.random:
        rng = np.random.default_rng(args.seed)
        index = int(rng.integers(0, len(dataset)))

    sample = dataset[index]
    sar_channels, optical_channels = sample["sar"].shape[2], sample["optical"].shape[2]
    print(f"Rendering {args.dataset}[{index}] (of {len(dataset)} total) through {args.checkpoint}...")

    device = torch.device(args.device)
    generator = load_generator(
        args.checkpoint, args.model, sar_channels, optical_channels, device,
        num_downs=args.num_downs, n_residual_blocks=args.n_residual_blocks,
    )

    fig = render_checkpoint_figure(
        sample["sar"], sample["optical"], generator, device, title_suffix=f" ({Path(args.checkpoint).name})"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
