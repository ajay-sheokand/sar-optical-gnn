"""
Tests for src/models/pix2pix.py's UNetGenerator.

Shape correctness across this project's three real patch sizes (112/120/256) is the main risk
here -- see the module's own docstring for why num_downs isn't auto-computed, and
src/models/blocks.py's match_spatial for how odd intermediate sizes are handled if they occur.
"""

import pytest
import torch

from src.models.pix2pix import UNetGenerator


class TestUNetGenerator:
    @pytest.mark.parametrize(
        "size,num_downs",
        [
            (112, 6),  # SARptical: 112->56->28->14->7->3->1 (Conv2d k4s2p1 floors odd sizes)
            (120, 6),  # BigEarthNet: 120->60->30->15->7->3->1
            (256, 7),  # SEN1-2: 256->128->64->32->16->8->4->2 (power of 2, halves exactly)
        ],
    )
    def test_output_matches_input_spatial_size_on_real_dataset_sizes(self, size, num_downs):
        generator = UNetGenerator(in_channels=2, out_channels=3, num_downs=num_downs)
        x = torch.randn(1, 2, size, size)
        out = generator(x)
        assert out.shape == (1, 3, size, size)

    @pytest.mark.parametrize("in_channels,out_channels", [(1, 3), (2, 3), (2, 12)])
    def test_handles_real_channel_combinations(self, in_channels, out_channels):
        """1->3: SARptical (1-channel SAR amplitude to RGB). 2->3: real Sentinel-1 VV+VH to RGB.
        2->12: real Sentinel-1 to full multispectral Sentinel-2."""
        generator = UNetGenerator(in_channels=in_channels, out_channels=out_channels, num_downs=5)
        x = torch.randn(2, in_channels, 96, 96)
        out = generator(x)
        assert out.shape == (2, out_channels, 96, 96)

    def test_output_is_tanh_bounded(self):
        generator = UNetGenerator(in_channels=2, out_channels=3, num_downs=5)
        out = generator(torch.randn(1, 2, 96, 96))
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_gradients_flow_to_every_parameter(self):
        generator = UNetGenerator(in_channels=2, out_channels=3, num_downs=5)
        out = generator(torch.randn(1, 2, 96, 96))
        out.sum().backward()
        for name, param in generator.named_parameters():
            assert param.grad is not None, f"no gradient reached {name}"

    def test_rejects_too_few_downsampling_steps(self):
        with pytest.raises(ValueError, match="num_downs"):
            UNetGenerator(in_channels=2, out_channels=3, num_downs=1)
