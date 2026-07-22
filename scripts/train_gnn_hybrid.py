#!/usr/bin/env python
"""
Train the M4 novel model: GraphHybridGenerator (src/models/gnn_hybrid.py), a GNN structural prior
over the SAR superpixel graph injected into the same pix2pix U-Net backbone M3's baseline uses.

Why this is a separate script from scripts/train_baseline.py rather than a third --model option
there: this model needs a genuinely different data pipeline (a cached graph per sample, loaded via
src/datasets/graph_dataset.py's GraphHybridDataset) and is currently batch-size-1 only (see
src/models/gnn_hybrid.py's module docstring for why), both real enough differences to justify a
dedicated script rather than branching train_baseline.py's shared training loop three ways.
Reuses what's already proven there rather than duplicating it: GANLoss, TranslationMetrics,
split_dataset, save_checkpoint, log_metrics, find_latest_checkpoint.

Usage:
    python -m scripts.train_gnn_hybrid --dataset sen1_2 --root data/sen1_2 \\
        --out outputs/gnn_hybrid_sen1_2 --epochs 40

    # The ablation this project's whole research question hinges on -- graph branch disabled,
    # which (per src/models/gnn_hybrid.py's docstring) collapses to the exact pix2pix architecture:
    python -m scripts.train_gnn_hybrid --dataset sen1_2 --root data/sen1_2 \\
        --out outputs/gnn_hybrid_no_graph --epochs 40 --no-graph-branch
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.build_graphs_offline import _load_dataset, cache_graph
from scripts.train_baseline import find_latest_checkpoint, log_metrics, save_checkpoint, split_dataset
from src.datasets.graph_dataset import GraphHybridDataset
from src.eval.metrics import TranslationMetrics
from src.models.blocks import PatchGANDiscriminator
from src.models.gnn_hybrid import GraphHybridGenerator
from src.models.losses import GANLoss


def ensure_graphs_cached(base_dataset, cache_dir: Path, num_segments: int) -> None:
    """Caches one graph per sample if it isn't already there -- idempotent, so re-running this
    script (e.g. after a Kaggle session restart) doesn't redo work. Uses the SAR channel only
    (build_graphs_offline.py's cache_graph signature), matching M2's design: the graph is a
    property of the SAR image's structure, not the (unknown, being-predicted) optical image."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index in range(len(base_dataset)):
        path = cache_dir / f"{index:07d}.npz"
        if path.exists():
            continue
        sample = base_dataset[index]
        cache_graph(sample["sar"], path, num_segments=num_segments)


def prune_checkpoints(out_dir: Path, keep_every: int, current_epoch: int) -> None:
    """Deletes old epoch_*.pt files except every `keep_every`th one and the just-saved current
    epoch. Needed on Kaggle specifically: each checkpoint here is ~550MB (U-Net generator +
    PatchGAN discriminator + both Adam optimizer states), and train_baseline.py's save_checkpoint
    never prunes -- fine for local disk, but an unpruned 80-epoch Kaggle run would hit the
    platform's 20GB /kaggle/working output cap around epoch 36, long before training finishes.
    Keeping every 5th epoch (the default) preserves a checkpoint history for
    scripts/watch_and_render.sh's visual spot-checks while keeping total disk bounded."""
    for path in out_dir.glob("epoch_*.pt"):
        epoch = int(path.stem.split("_")[1])
        if epoch % keep_every == 0 or epoch == current_epoch:
            continue
        path.unlink()


def infer_dims(dataset):
    """Peek at one sample to get the channel/feature-vector dimensions GraphHybridGenerator's
    constructor needs fixed at build time (same reasoning as train_baseline.py's
    infer_channel_counts)."""
    sample = dataset[0]
    sar_channels = sample["sar"].shape[0]
    optical_channels = sample["optical"].shape[0]
    node_feature_dim = sample["node_features"].shape[1]
    return sar_channels, optical_channels, node_feature_dim


def gnn_hybrid_train_step(generator, discriminator, opt_g, opt_d, gan_loss, batch, lambda_l1, lambda_node_aux, device):
    """One training step. `batch` comes from a batch_size=1 DataLoader (see this module's
    docstring on why batch size is 1) -- sar/optical keep their DataLoader-added batch dimension
    (matches what GraphHybridGenerator.forward expects), the graph tensors have theirs squeezed
    back off (the model consumes one graph directly, not a length-1 batch of graphs)."""
    sar = batch["sar"].to(device)
    optical = batch["optical"].to(device)
    node_features = batch["node_features"][0].to(device)
    edge_index = batch["edge_index"][0].to(device)
    label_map = batch["label_map"][0].to(device)
    node_targets = batch["node_targets"][0].to(device)

    fake_optical, node_predictions = generator(sar, node_features, edge_index, label_map)

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
    loss_node_aux = (
        F.l1_loss(node_predictions, node_targets)
        if node_predictions is not None
        else torch.tensor(0.0, device=device)
    )
    loss_g = loss_g_adv + lambda_l1 * loss_g_l1 + lambda_node_aux * loss_node_aux
    loss_g.backward()
    opt_g.step()

    return {
        "loss_d": loss_d.item(),
        "loss_g": loss_g.item(),
        "loss_g_adv": loss_g_adv.item(),
        "loss_g_l1": loss_g_l1.item(),
        "loss_node_aux": loss_node_aux.item() if node_predictions is not None else 0.0,
    }


@torch.no_grad()
def evaluate_gnn_hybrid(generator, val_loader, device, metrics):
    """Reuses TranslationMetrics the same reset()-not-reconstructed way train_baseline.py's
    evaluate_pix2pix does -- see that function's docstring for the real GPU-memory leak this
    avoids (docs/BUILD_LOG.md's M3 entry)."""
    metrics.reset()
    generator.eval()
    for batch in val_loader:
        sar = batch["sar"].to(device)
        optical = batch["optical"].to(device)
        node_features = batch["node_features"][0].to(device) if generator.use_graph_branch else None
        edge_index = batch["edge_index"][0].to(device) if generator.use_graph_branch else None
        label_map = batch["label_map"][0].to(device) if generator.use_graph_branch else None

        fake_optical, _ = generator(sar, node_features, edge_index, label_map)
        metrics.update_pixel_metrics(fake_optical, optical)
        if optical.shape[1] == 3:
            metrics.update_fid(fake_optical, optical)
    generator.train()
    return metrics.compute()


def train_gnn_hybrid(dataset, args, device):
    sar_channels, optical_channels, node_feature_dim = infer_dims(dataset)
    generator = GraphHybridGenerator(
        sar_channels, optical_channels, node_feature_dim,
        num_downs=args.num_downs, gnn_hidden_dim=args.gnn_hidden_dim, gnn_layers=args.gnn_layers,
        use_graph_branch=not args.no_graph_branch,
    ).to(device)
    discriminator = PatchGANDiscriminator(sar_channels + optical_channels).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    gan_loss = GANLoss().to(device)
    metrics = TranslationMetrics(device=device)

    train_set, val_set = split_dataset(dataset, args.val_fraction, args.seed)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers)

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
            print(f"[gnn_hybrid] resumed from {latest} (completed through epoch {checkpoint['epoch']})")

    if start_epoch > args.epochs:
        print(f"[gnn_hybrid] already completed {start_epoch - 1}/{args.epochs} epochs -- nothing to do")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        epoch_losses = []
        for batch in train_loader:
            losses = gnn_hybrid_train_step(
                generator, discriminator, opt_g, opt_d, gan_loss, batch,
                args.lambda_l1, args.lambda_node_aux, device,
            )
            epoch_losses.append(losses)

        mean_losses = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) for k in epoch_losses[0]}
        val_metrics = evaluate_gnn_hybrid(generator, val_loader, device, metrics)
        elapsed = time.perf_counter() - start

        record = {"epoch": epoch, "elapsed_seconds": elapsed, "train_loss": mean_losses, "val_metrics": val_metrics}
        print(f"[gnn_hybrid] epoch {epoch}/{args.epochs} ({elapsed:.1f}s) train={mean_losses} val={val_metrics}")
        log_metrics(out_dir, record)
        save_checkpoint({
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch": epoch,
        }, out_dir, epoch)
        if args.keep_every_n_checkpoints > 0:
            prune_checkpoints(out_dir, args.keep_every_n_checkpoints, epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["bigearthnet", "sen12ms", "sen1_2", "sarptical"], required=True)
    parser.add_argument("--root", required=True, help="dataset root directory (must already be downloaded)")
    parser.add_argument("--out", required=True, help="output directory for checkpoints + metrics.jsonl")
    parser.add_argument("--graph-cache", default=None, help="directory for cached graphs (default: <out>/graph_cache)")
    parser.add_argument("--num-segments", type=int, default=100, help="SLIC target superpixel count")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-downs", type=int, default=6, help="wrapped U-Net depth")
    parser.add_argument("--gnn-hidden-dim", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--no-graph-branch", action="store_true", help="ablation: disable the graph branch entirely")
    parser.add_argument("--lambda-l1", type=float, default=100.0, help="pixel L1 weight (matches pix2pix's default)")
    parser.add_argument("--lambda-node-aux", type=float, default=10.0, help="node-auxiliary L1 weight")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N samples (smoke-testing)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints in --out")
    parser.add_argument(
        "--keep-every-n-checkpoints", type=int, default=5,
        help="prune epoch_*.pt files except every Nth epoch and the latest (0 disables pruning)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # Raw dataset, not wrapped in TranslationDataset (src/datasets/adapter.py) -- GraphHybridDataset
    # does its own normalization at the right point (after pooling raw pixel values for the
    # node-auxiliary target, see its docstring), so the extra normalization layer isn't wanted here.
    base_dataset = _load_dataset(args.dataset, args.root)
    if args.limit is not None:
        base_dataset = torch.utils.data.Subset(base_dataset, range(min(args.limit, len(base_dataset))))

    graph_cache_dir = Path(args.graph_cache) if args.graph_cache else Path(args.out) / "graph_cache"
    print(f"Ensuring graphs are cached at {graph_cache_dir} ({len(base_dataset)} samples)...")
    ensure_graphs_cached(base_dataset, graph_cache_dir, args.num_segments)

    dataset = GraphHybridDataset(base_dataset, graph_cache_dir)
    device = torch.device(args.device)
    train_gnn_hybrid(dataset, args, device)


if __name__ == "__main__":
    main()
