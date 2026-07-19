"""
Shared building blocks for the M3 baseline generators/discriminators (pix2pix, CycleGAN).

Why these are pulled out into one file: pix2pix's U-Net generator and CycleGAN's ResNet generator
both eventually feed the same kind of discriminator (a PatchGAN — see docs/RESEARCH_PLAN.md §5),
and this project's three real datasets have three different, non-power-of-2-friendly patch sizes
(SARptical 112px, BigEarthNet 120px, SEN1-2 256px — see src/datasets/*.py docstrings). None of
these are guaranteed to survive a naive encoder/decoder size round-trip without an explicit
size-matching step at the skip connections, so that logic (`match_spatial`) lives here once
instead of being duplicated or silently assumed away in each generator.
"""

import torch
import torch.nn as nn


def match_spatial(x, reference):
    """
    Center-crop or zero-pad `x` so its spatial dims (H, W) match `reference`'s.

    Why this exists: a U-Net's skip connections concatenate an upsampled decoder feature map with
    the corresponding encoder feature map. `nn.Conv2d(kernel=4, stride=2, padding=1)` on an
    odd-sized input floors the output size (e.g. 15 -> 7), while `nn.ConvTranspose2d(kernel=4,
    stride=2, padding=1)` on that same 7 always doubles it back to exactly 14, not 15 — a 1-pixel
    mismatch that torch.cat will refuse to concatenate. All three of this project's actual dataset
    patch sizes (112, 120, 256) are even and happen to round-trip exactly through
    UNetGenerator's chosen down/up settings, so this function is a no-op in practice today — it's
    kept as an explicit guard (with a test covering odd sizes directly) rather than a silent
    assumption, since "the sizes we currently use happen to divide evenly" is exactly the kind of
    fact that breaks quietly if a new dataset or patch size is added later.
    """
    _, _, h, w = reference.shape
    _, _, xh, xw = x.shape

    if xh > h:
        top = (xh - h) // 2
        x = x[:, :, top : top + h, :]
    elif xh < h:
        pad_top = (h - xh) // 2
        pad_bottom = h - xh - pad_top
        x = nn.functional.pad(x, (0, 0, pad_top, pad_bottom))

    if xw > w:
        left = (xw - w) // 2
        x = x[:, :, :, left : left + w]
    elif xw < w:
        pad_left = (w - xw) // 2
        pad_right = w - xw - pad_left
        x = nn.functional.pad(x, (pad_left, pad_right, 0, 0))

    return x


class UNetDownBlock(nn.Module):
    """One encoder step of the pix2pix U-Net: strided conv halves H and W."""

    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUpBlock(nn.Module):
    """One decoder step of the pix2pix U-Net: transposed conv doubles H and W, then concatenates
    the matching encoder skip connection (see match_spatial's docstring for why sizes are checked
    rather than assumed)."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        x = match_spatial(x, skip)
        return torch.cat([x, skip], dim=1)


class ResidualBlock(nn.Module):
    """CycleGAN's ResNet generator building block: two 3x3 convs with a skip connection, reflect
    padding (avoids the border artifacts zero-padding introduces over repeated residual blocks)."""

    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.model(x)


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator (Isola et al. 2017 / Zhu et al. 2017), shared by both pix2pix
    and CycleGAN. Classifies overlapping patches of the input as real/fake rather than the whole
    image at once, which is what lets it work on any input resolution large enough for its
    downsampling stages -- it's fully convolutional, with no fully-connected layer that would
    hardcode an expected input size. In practice it needs input dimensions well above 2^n_layers
    pixels per side, since InstanceNorm2d refuses a collapsed 1x1 spatial input (this project's
    three real patch sizes, 112/120/256, are comfortably above that floor with the default
    n_layers=3; synthetic test images as small as 16x16 are not). `in_channels` is set
    by the caller: pix2pix concatenates the SAR condition with the optical image before feeding
    this (in_channels = sar_channels + optical_channels), CycleGAN feeds a single domain's image
    alone (in_channels = that domain's channel count).
    """

    def __init__(self, in_channels, ndf=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = ndf
        for i in range(1, n_layers):
            next_channels = min(channels * 2, ndf * 8)
            layers += [
                nn.Conv2d(channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels = next_channels

        next_channels = min(channels * 2, ndf * 8)
        layers += [
            nn.Conv2d(channels, next_channels, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(next_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers.append(nn.Conv2d(next_channels, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
