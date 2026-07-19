"""
CycleGAN generator — ResNet encoder/residual-blocks/decoder (Zhu et al. 2017), the second M3
baseline. See docs/RESEARCH_PLAN.md §5: CycleGAN is *unpaired* (no direct SAR<->optical
correspondence required, trained via cycle-consistency instead of pixel loss), which is why it
doubles as an ablation on this project's own paired datasets -- comparing it against pix2pix
(same datasets, same discriminator design, but paired L1 supervision) isolates how much paired
supervision itself is worth, separate from any graph-based structural prior added later in M4.

Unlike pix2pix's U-Net (src/models/pix2pix.py), this generator has no skip connections between
its encoder and decoder halves -- it's a single downsample -> residual-blocks -> upsample chain,
so src/models/blocks.py's match_spatial isn't needed here: with kernel=3/stride=2/padding=1 down
and kernel=3/stride=2/padding=1/output_padding=1 up, every downsample-then-upsample pair returns
to *exactly* the original spatial size for any even input dimension (verified for this project's
three real patch sizes: 112, 120, 256 -- see tests/models/test_cyclegan.py).
"""

import torch.nn as nn

from src.models.blocks import ResidualBlock


class ResnetGenerator(nn.Module):
    """
    Args:
        in_channels: source-domain channel count.
        out_channels: target-domain channel count. CycleGAN trains two of these (SAR->optical and
            optical->SAR) for the cycle-consistency loss, so in/out are swapped between them.
        n_residual_blocks: 9 for high-resolution training (the original paper's setting for
            256x256+ images -- matches this project's SEN1-2 patches), 6 is the paper's own choice
            for smaller images; exposed as a parameter rather than hardcoded so the training script
            can pick per-dataset.
        ngf: base channel count after the initial conv, matching pix2pix's `ngf` in spirit (not
            directly comparable in value, since this architecture's channel schedule is different).
    """

    def __init__(self, in_channels, out_channels, n_residual_blocks=9, ngf=64):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        channels = ngf
        for _ in range(2):
            next_channels = channels * 2
            layers += [
                nn.Conv2d(channels, next_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(next_channels),
                nn.ReLU(inplace=True),
            ]
            channels = next_channels

        for _ in range(n_residual_blocks):
            layers.append(ResidualBlock(channels))

        for _ in range(2):
            next_channels = channels // 2
            layers += [
                nn.ConvTranspose2d(channels, next_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(next_channels),
                nn.ReLU(inplace=True),
            ]
            channels = next_channels

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
