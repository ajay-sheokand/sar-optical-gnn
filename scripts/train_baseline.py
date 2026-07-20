#!/usr/bin/env python
"""
Train the M3 baselines (pix2pix, CycleGAN) on any of this project's dataset loaders.

Why this script exists (docs/RESEARCH_PLAN.md §7, M3): before the novel graph-hybrid model (M4)
gets built, this project needs its own faithful pix2pix and CycleGAN reimplementations trained
and evaluated, primarily so the SEN1-2 validation pass (§4) can check whether the resulting
PSNR/SSIM/FID land near already-published numbers -- if they do, the data pipeline, training
loop, and metrics code (src/eval/metrics.py) are trustworthy before anything novel is layered on
top of them. The same code trains on any dataset this project has a loader for (see
scripts/build_graphs_offline.py's _load_dataset, reused here rather than duplicated).

Usage:
    python scripts/train_baseline.py --model pix2pix --dataset sarptical \\
        --root data/sarptical/patch_SAR_OPT_SQUARE --out outputs/pix2pix_sarptical --epochs 20

    python scripts/train_baseline.py --model cyclegan --dataset sen1_2 \\
        --root data/sen1_2 --out outputs/cyclegan_sen1_2 --epochs 20 --n-residual-blocks 9

pix2pix trains a single U-Net generator (src/models/pix2pix.py) conditioned on SAR, supervised
with paired L1 + adversarial loss. CycleGAN trains two ResNet generators (src/models/cyclegan.py)
in both directions with cycle-consistency + adversarial loss, ignoring the SAR/optical pairing
during training (by design -- see src/models/cyclegan.py's docstring) but still *evaluated*
against the real paired ground truth, same as pix2pix, since every dataset this project uses
happens to have real pairs available for that check.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.datasets.adapter import TranslationDataset
from src.eval.metrics import TranslationMetrics
from src.models.blocks import PatchGANDiscriminator
from src.models.cyclegan import ResnetGenerator
from src.models.losses import GANLoss, cycle_consistency_loss, identity_loss
from src.models.pix2pix import UNetGenerator


def load_dataset(name: str, root: str) -> TranslationDataset:
    """Dispatch to a real loader (src/datasets/*) and wrap it model-ready. Reuses
    scripts/build_graphs_offline.py's dispatch rather than duplicating the same if/elif chain."""
    from scripts.build_graphs_offline import _load_dataset

    return TranslationDataset(_load_dataset(name, root))


def split_dataset(dataset, val_fraction: float = 0.1, seed: int = 0):
    """Train/val split. `val_fraction` of the dataset (at least 1 sample) goes to validation --
    val is what the SEN1-2-style PSNR/SSIM/FID check (§4) is actually computed against, so it must
    never be empty, even for a tiny --limit'd smoke run."""
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    if n_train < 1:
        raise ValueError(f"dataset of size {len(dataset)} is too small for a val_fraction={val_fraction} split")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def infer_channel_counts(dataset) -> tuple[int, int]:
    """Peek at one sample to determine SAR/optical channel counts -- these vary by dataset (SAR:
    1 for SARptical, 2 for real Sentinel-1; optical: 3 for RGB, up to 12+ for full multispectral),
    and src/models/ generators need them fixed at construction time (see UNetGenerator's
    docstring for why this can't be inferred lazily at first forward() call instead)."""
    sample = dataset[0]
    return sample["sar"].shape[0], sample["optical"].shape[0]


# --- pix2pix ---------------------------------------------------------------------------------


def build_pix2pix(sar_channels: int, optical_channels: int, num_downs: int, device):
    generator = UNetGenerator(sar_channels, optical_channels, num_downs=num_downs).to(device)
    discriminator = PatchGANDiscriminator(sar_channels + optical_channels).to(device)
    return generator, discriminator


def pix2pix_train_step(generator, discriminator, opt_g, opt_d, gan_loss, sar, optical, lambda_l1, device):
    sar, optical = sar.to(device), optical.to(device)
    fake_optical = generator(sar)

    opt_d.zero_grad()
    real_pred = discriminator(torch.cat([sar, optical], dim=1))
    fake_pred = discriminator(torch.cat([sar, fake_optical.detach()], dim=1))
    loss_d = 0.5 * (gan_loss(real_pred, True) + gan_loss(fake_pred, False))
    loss_d.backward()
    opt_d.step()

    opt_g.zero_grad()
    fake_pred_for_g = discriminator(torch.cat([sar, fake_optical], dim=1))
    loss_g_adv = gan_loss(fake_pred_for_g, True)
    loss_g_l1 = F.l1_loss(fake_optical, optical)
    loss_g = loss_g_adv + lambda_l1 * loss_g_l1
    loss_g.backward()
    opt_g.step()

    return {
        "loss_d": loss_d.item(),
        "loss_g": loss_g.item(),
        "loss_g_adv": loss_g_adv.item(),
        "loss_g_l1": loss_g_l1.item(),
    }


@torch.no_grad()
def evaluate_pix2pix(generator, val_loader, device, metrics):
    """`metrics` is a caller-owned TranslationMetrics instance, reset here rather than
    constructed fresh each call -- see docs/BUILD_LOG.md's M3 entry: FrechetInceptionDistance
    (inside TranslationMetrics) forms a Python reference cycle that ordinary refcounting never
    collects, so building a new one every epoch leaked ~230MB/epoch and OOM'd a real training run
    after 26 epochs. Reusing one instance for the whole training run sidesteps the leak instead of
    papering over it with a per-epoch gc.collect()."""
    metrics.reset()
    generator.eval()
    for batch in val_loader:
        sar, optical = batch["sar"].to(device), batch["optical"].to(device)
        fake_optical = generator(sar)
        metrics.update_pixel_metrics(fake_optical, optical)
        if optical.shape[1] == 3:
            metrics.update_fid(fake_optical, optical)
    generator.train()
    return metrics.compute()


# --- CycleGAN ----------------------------------------------------------------------------------


def build_cyclegan(sar_channels: int, optical_channels: int, n_residual_blocks: int, device):
    g_ab = ResnetGenerator(sar_channels, optical_channels, n_residual_blocks=n_residual_blocks).to(device)
    g_ba = ResnetGenerator(optical_channels, sar_channels, n_residual_blocks=n_residual_blocks).to(device)
    d_a = PatchGANDiscriminator(sar_channels).to(device)
    d_b = PatchGANDiscriminator(optical_channels).to(device)
    return g_ab, g_ba, d_a, d_b


def cyclegan_train_step(
    g_ab, g_ba, d_a, d_b, opt_g, opt_d, gan_loss, sar, optical, lambda_cycle, lambda_identity, device
):
    sar, optical = sar.to(device), optical.to(device)
    use_identity = lambda_identity > 0 and sar.shape[1] == optical.shape[1]

    opt_g.zero_grad()
    fake_optical = g_ab(sar)
    fake_sar = g_ba(optical)
    recovered_sar = g_ba(fake_optical)
    recovered_optical = g_ab(fake_sar)

    loss_gan = gan_loss(d_b(fake_optical), True) + gan_loss(d_a(fake_sar), True)
    loss_cycle = cycle_consistency_loss(recovered_sar, sar) + cycle_consistency_loss(recovered_optical, optical)
    loss_g = loss_gan + lambda_cycle * loss_cycle

    loss_identity_value = 0.0
    if use_identity:
        # identity_loss checks G(y) ~= y for y already in G's *output* domain, which only
        # type-checks when a generator's in/out channel counts match (see cyclegan.py's module
        # docstring) -- true for e.g. a same-channel-count ablation, not for SAR(1-2ch)->
        # optical(3ch+) by default, so this is skipped unless the caller's channel counts allow it.
        loss_identity = identity_loss(g_ab(optical), optical) + identity_loss(g_ba(sar), sar)
        loss_g = loss_g + lambda_identity * loss_identity
        loss_identity_value = loss_identity.item()

    loss_g.backward()
    opt_g.step()

    opt_d.zero_grad()
    loss_d_a = 0.5 * (gan_loss(d_a(sar), True) + gan_loss(d_a(fake_sar.detach()), False))
    loss_d_b = 0.5 * (gan_loss(d_b(optical), True) + gan_loss(d_b(fake_optical.detach()), False))
    loss_d = loss_d_a + loss_d_b
    loss_d.backward()
    opt_d.step()

    return {
        "loss_d": loss_d.item(),
        "loss_g": loss_g.item(),
        "loss_gan": loss_gan.item(),
        "loss_cycle": loss_cycle.item(),
        "loss_identity": loss_identity_value,
    }


@torch.no_grad()
def evaluate_cyclegan(g_ab, val_loader, device, metrics):
    """Evaluates only the SAR->optical direction against real paired ground truth -- the
    direction comparable to pix2pix's output, even though CycleGAN never used the pairing during
    training (see this module's docstring). `metrics` is caller-owned and reset here, not
    constructed fresh -- see evaluate_pix2pix's docstring for why (a real GPU-memory leak this
    caused in a real training run)."""
    metrics.reset()
    g_ab.eval()
    for batch in val_loader:
        sar, optical = batch["sar"].to(device), batch["optical"].to(device)
        fake_optical = g_ab(sar)
        metrics.update_pixel_metrics(fake_optical, optical)
        if optical.shape[1] == 3:
            metrics.update_fid(fake_optical, optical)
    g_ab.train()
    return metrics.compute()


# --- shared training driver ---------------------------------------------------------------------


def save_checkpoint(state, out_dir: Path, epoch: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"epoch_{epoch:04d}.pt"
    torch.save(state, path)
    return path


def find_latest_checkpoint(out_dir) -> Path | None:
    """Returns the highest-epoch `epoch_NNNN.pt` file under `out_dir`, or None if there isn't
    one. Filenames sort correctly as strings (zero-padded to 4 digits), so a plain sort works
    without needing to parse the epoch number out first.

    Why this exists: Kaggle GPU sessions are capped at roughly 9-12 continuous hours, and this
    project's real SEN1-2 run took ~19 hours combined (docs/BUILD_LOG.md's M3 entry) -- a session
    that gets cut off mid-run needs a way to pick back up without redoing already-completed
    epochs. This is also exactly what would have saved the ~68 minutes of progress lost when a
    real local run OOM'd at epoch 27 before the memory leak fix (same doc, §9b).
    """
    checkpoints = sorted(Path(out_dir).glob("epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def log_metrics(out_dir: Path, record: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def train_pix2pix(dataset, args, device):
    train_set, val_set = split_dataset(dataset, args.val_fraction, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    sar_channels, optical_channels = infer_channel_counts(dataset)
    generator, discriminator = build_pix2pix(sar_channels, optical_channels, args.num_downs, device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    gan_loss = GANLoss().to(device)
    # Constructed once, reused every epoch (evaluate_pix2pix resets it) -- see its docstring for
    # why: a fresh TranslationMetrics per epoch leaked GPU memory via FrechetInceptionDistance's
    # reference cycle and OOM'd a real training run after 26 epochs.
    metrics = TranslationMetrics(device=device)

    out_dir = Path(args.out)
    start_epoch = 1
    if not args.no_resume:
        latest = find_latest_checkpoint(out_dir)
        if latest is not None:
            checkpoint = torch.load(latest, map_location=device, weights_only=True)
            generator.load_state_dict(checkpoint["generator"])
            discriminator.load_state_dict(checkpoint["discriminator"])
            opt_g.load_state_dict(checkpoint["opt_g"])
            opt_d.load_state_dict(checkpoint["opt_d"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"[pix2pix] resumed from {latest} (completed through epoch {checkpoint['epoch']})")

    if start_epoch > args.epochs:
        print(f"[pix2pix] already completed {start_epoch - 1}/{args.epochs} epochs -- nothing to do")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        epoch_losses = []
        for batch in train_loader:
            losses = pix2pix_train_step(
                generator, discriminator, opt_g, opt_d, gan_loss, batch["sar"], batch["optical"], args.lambda_l1, device
            )
            epoch_losses.append(losses)

        mean_losses = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) for k in epoch_losses[0]}
        val_metrics = evaluate_pix2pix(generator, val_loader, device, metrics)
        elapsed = time.perf_counter() - start

        record = {"epoch": epoch, "elapsed_seconds": elapsed, "train_loss": mean_losses, "val_metrics": val_metrics}
        print(f"[pix2pix] epoch {epoch}/{args.epochs} ({elapsed:.1f}s) train={mean_losses} val={val_metrics}")
        log_metrics(out_dir, record)
        save_checkpoint({
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch": epoch,
        }, out_dir, epoch)


def train_cyclegan(dataset, args, device):
    train_set, val_set = split_dataset(dataset, args.val_fraction, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    sar_channels, optical_channels = infer_channel_counts(dataset)
    g_ab, g_ba, d_a, d_b = build_cyclegan(sar_channels, optical_channels, args.n_residual_blocks, device)
    opt_g = torch.optim.Adam(
        list(g_ab.parameters()) + list(g_ba.parameters()), lr=args.lr, betas=(0.5, 0.999)
    )
    opt_d = torch.optim.Adam(list(d_a.parameters()) + list(d_b.parameters()), lr=args.lr, betas=(0.5, 0.999))
    gan_loss = GANLoss().to(device)
    metrics = TranslationMetrics(device=device)  # see train_pix2pix's comment on why this is reused

    out_dir = Path(args.out)
    start_epoch = 1
    if not args.no_resume:
        latest = find_latest_checkpoint(out_dir)
        if latest is not None:
            checkpoint = torch.load(latest, map_location=device, weights_only=True)
            g_ab.load_state_dict(checkpoint["g_ab"])
            g_ba.load_state_dict(checkpoint["g_ba"])
            d_a.load_state_dict(checkpoint["d_a"])
            d_b.load_state_dict(checkpoint["d_b"])
            opt_g.load_state_dict(checkpoint["opt_g"])
            opt_d.load_state_dict(checkpoint["opt_d"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"[cyclegan] resumed from {latest} (completed through epoch {checkpoint['epoch']})")

    if start_epoch > args.epochs:
        print(f"[cyclegan] already completed {start_epoch - 1}/{args.epochs} epochs -- nothing to do")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        epoch_losses = []
        for batch in train_loader:
            losses = cyclegan_train_step(
                g_ab, g_ba, d_a, d_b, opt_g, opt_d, gan_loss,
                batch["sar"], batch["optical"], args.lambda_cycle, args.lambda_identity, device,
            )
            epoch_losses.append(losses)

        mean_losses = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) for k in epoch_losses[0]}
        val_metrics = evaluate_cyclegan(g_ab, val_loader, device, metrics)
        elapsed = time.perf_counter() - start

        record = {"epoch": epoch, "elapsed_seconds": elapsed, "train_loss": mean_losses, "val_metrics": val_metrics}
        print(f"[cyclegan] epoch {epoch}/{args.epochs} ({elapsed:.1f}s) train={mean_losses} val={val_metrics}")
        log_metrics(out_dir, record)
        save_checkpoint({
            "g_ab": g_ab.state_dict(),
            "g_ba": g_ba.state_dict(),
            "d_a": d_a.state_dict(),
            "d_b": d_b.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch": epoch,
        }, out_dir, epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["pix2pix", "cyclegan"], required=True)
    parser.add_argument("--dataset", choices=["bigearthnet", "sen12ms", "sen1_2", "sarptical"], required=True)
    parser.add_argument("--root", required=True, help="dataset root directory (must already be downloaded)")
    parser.add_argument("--out", required=True, help="output directory for checkpoints + metrics.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-downs", type=int, default=6, help="pix2pix U-Net depth")
    parser.add_argument("--n-residual-blocks", type=int, default=9, help="CycleGAN ResNet depth")
    parser.add_argument("--lambda-l1", type=float, default=100.0, help="pix2pix L1 weight (paper default)")
    parser.add_argument("--lambda-cycle", type=float, default=10.0, help="CycleGAN cycle-consistency weight (paper default)")
    parser.add_argument("--lambda-identity", type=float, default=0.0, help="CycleGAN identity weight; 0 disables it")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N samples (smoke-testing)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-resume", action="store_true",
        help="ignore any existing checkpoints in --out and start from epoch 1 (default: auto-resume "
        "from the latest epoch_NNNN.pt found there, if any -- see find_latest_checkpoint)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = load_dataset(args.dataset, args.root)
    if args.limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))

    device = torch.device(args.device)
    if args.model == "pix2pix":
        train_pix2pix(dataset, args, device)
    else:
        train_cyclegan(dataset, args, device)


if __name__ == "__main__":
    main()
