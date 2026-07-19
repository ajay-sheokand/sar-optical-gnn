"""
Tests for src/models/cyclegan.py's ResnetGenerator.

The key property this architecture claims over pix2pix's U-Net (see the module docstring): no
skip connections, so no match_spatial cropping/padding is needed -- the down/up conv settings are
chosen to round-trip to *exactly* the original spatial size for any even input. That round-trip
claim is checked directly here, not just output-shape-looks-plausible.
"""

import pytest
import torch

from src.models.cyclegan import ResnetGenerator


class TestResnetGenerator:
    @pytest.mark.parametrize("size", [112, 120, 256])
    def test_output_exactly_matches_input_spatial_size(self, size):
        """112 (SARptical), 120 (BigEarthNet), 256 (SEN1-2) -- all even, all should round-trip
        exactly through 2 downsamples + 2 upsamples with no cropping/padding needed."""
        generator = ResnetGenerator(in_channels=3, out_channels=3, n_residual_blocks=2)
        x = torch.randn(1, 3, size, size)
        out = generator(x)
        assert out.shape == (1, 3, size, size)

    @pytest.mark.parametrize("in_channels,out_channels", [(1, 3), (2, 3), (3, 2), (3, 1)])
    def test_handles_real_channel_combinations_in_both_directions(self, in_channels, out_channels):
        """CycleGAN trains two generators (SAR->optical and optical->SAR), so both directions of
        an asymmetric channel count need to work, not just SAR->optical."""
        generator = ResnetGenerator(in_channels=in_channels, out_channels=out_channels, n_residual_blocks=2)
        out = generator(torch.randn(1, in_channels, 96, 96))
        assert out.shape == (1, out_channels, 96, 96)

    def test_output_is_tanh_bounded(self):
        generator = ResnetGenerator(in_channels=3, out_channels=3, n_residual_blocks=2)
        out = generator(torch.randn(1, 3, 96, 96))
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_gradients_flow_to_every_parameter(self):
        generator = ResnetGenerator(in_channels=3, out_channels=3, n_residual_blocks=2)
        out = generator(torch.randn(1, 3, 96, 96))
        out.sum().backward()
        for name, param in generator.named_parameters():
            assert param.grad is not None, f"no gradient reached {name}"

    def test_more_residual_blocks_means_more_parameters(self):
        small = ResnetGenerator(in_channels=3, out_channels=3, n_residual_blocks=2)
        large = ResnetGenerator(in_channels=3, out_channels=3, n_residual_blocks=9)
        count = lambda m: sum(p.numel() for p in m.parameters())
        assert count(large) > count(small)
