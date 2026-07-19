"""
pix2pix generator — U-Net with skip connections (Isola et al. 2017), the first M3 baseline.

pix2pix is a *paired* conditional GAN: the generator is conditioned directly on the SAR image
(not a random noise vector) and trained against ground-truth optical via a combination of L1
pixel loss and adversarial loss from src/models/blocks.py's PatchGANDiscriminator. See
docs/RESEARCH_PLAN.md §5 for why this is one of only two baselines this project reimplements
(vs. cites) — it's the standard nearly the whole SAR-optical-translation field builds on, and
having our own faithful reimplementation is what lets the SEN1-2 validation pass (§4) actually
mean something: if our numbers land near the literature's, the data/training/metrics stack is
trustworthy before the novel graph-hybrid model (M4) is built on top of it.
"""

import torch.nn as nn

from src.models.blocks import UNetDownBlock, UNetUpBlock


class UNetGenerator(nn.Module):
    """
    Encoder-decoder with skip connections at every resolution level (the actual "U" shape this
    architecture is named for) -- skip connections matter here specifically because SAR-to-optical
    translation is meant to preserve structure/geometry from the input, not hallucinate it from a
    compressed bottleneck alone, which is the same motivation given for skip connections in
    docs/RESEARCH_PLAN.md §8 risk 2 (structural fidelity over raw compression).

    Args:
        in_channels: SAR channel count (1 for SARptical amplitude, 2 for real Sentinel-1 VV+VH).
        out_channels: optical channel count (3 for RGB, up to 12+ for full multispectral).
        num_downs: number of downsampling steps. Must be chosen so the bottleneck doesn't collapse
            below spatial size 1 -- e.g. 6 works for this project's smallest patch size (112px:
            112->56->28->14->7->3->1, bottleneck lands exactly at 1x1, which is why the deepest
            down-block skips InstanceNorm -- see the constructor). Deliberately not auto-computed
            from input size, because nn.Module layers must be fixed at construction time; the
            caller (the training script) is expected to pick this per-dataset and it's cheap to
            get wrong loudly (shape mismatch) rather than silently.
        ngf: base channel count for the outermost conv layer; doubles at each down step, capped
            at ngf*8 (standard pix2pix choice, avoids the channel count exploding on deep inputs
            like SEN1-2's 256px patches).
        dropout: applied to the innermost 3 up-blocks only, matching the original pix2pix paper's
            use of dropout as a stand-in for the injected-noise vector conditional GANs otherwise
            need (this generator has no separate noise input).
    """

    def __init__(self, in_channels, out_channels, num_downs=6, ngf=64, dropout=0.5):
        super().__init__()
        if num_downs < 2:
            raise ValueError(f"num_downs must be >= 2 to have a meaningful bottleneck, got {num_downs}")

        down_channels = [ngf * min(2**i, 8) for i in range(num_downs)]

        self.downs = nn.ModuleList()
        prev_channels = in_channels
        for i, out_ch in enumerate(down_channels):
            # InstanceNorm2d refuses a 1x1 spatial input (nothing to normalize over per-channel),
            # which the deepest down-block often produces once the bottleneck gets small enough
            # (e.g. 112px with num_downs=6 bottlenecks at 1x1 -- see tests/models/test_pix2pix.py).
            # The outermost block (i==0) also skips normalization, matching the original pix2pix
            # reference architecture (the raw input shouldn't be instance-normalized).
            normalize = i not in (0, num_downs - 1)
            self.downs.append(UNetDownBlock(prev_channels, out_ch, normalize=normalize))
            prev_channels = out_ch

        # There are num_downs halvings on the way down, so num_downs doublings are needed on the
        # way back up. Only the first (num_downs - 1) of those have a same-resolution encoder
        # feature map to concatenate against (down_channels[:-1], deepest first) -- the very last
        # doubling returns to the *original* input resolution, which the encoder never stored a
        # feature map for (down_channels[0] is already at half resolution), so it has no skip
        # connection and is handled separately as self.final below.
        self.ups = nn.ModuleList()
        up_out_channels = list(reversed(down_channels[:-1]))
        for i, out_ch in enumerate(up_out_channels):
            use_dropout = dropout > 0 and i < 3
            self.ups.append(UNetUpBlock(prev_channels, out_ch, dropout=dropout if use_dropout else 0.0))
            prev_channels = out_ch * 2  # concatenated with the skip connection

        self.final = nn.Sequential(
            nn.ConvTranspose2d(prev_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # The innermost down-block's output is the bottleneck itself, not a skip connection for
        # any up-block (there's nothing "above" it to skip to) -- the up-path uses the remaining
        # skips in reverse order.
        bottleneck = skips.pop()
        x = bottleneck
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.final(x)
