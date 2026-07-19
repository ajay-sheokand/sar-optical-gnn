"""
Tests for src/models/blocks.py -- the shared conv building blocks pix2pix and CycleGAN sit on
top of. Focused on shape contracts, since a silent shape mismatch here (e.g. match_spatial's
crop/pad logic being off by one) would corrupt every downstream generator without necessarily
crashing loudly.
"""

import torch

from src.models.blocks import (
    PatchGANDiscriminator,
    ResidualBlock,
    UNetDownBlock,
    UNetUpBlock,
    match_spatial,
)


class TestMatchSpatial:
    def test_noop_when_already_matching(self):
        x = torch.randn(2, 4, 16, 16)
        ref = torch.randn(2, 4, 16, 16)
        out = match_spatial(x, ref)
        assert torch.equal(out, x)

    def test_crops_when_larger(self):
        x = torch.randn(2, 4, 17, 17)
        ref = torch.randn(2, 4, 15, 15)
        out = match_spatial(x, ref)
        assert out.shape[-2:] == (15, 15)

    def test_pads_when_smaller(self):
        x = torch.randn(2, 4, 13, 13)
        ref = torch.randn(2, 4, 15, 15)
        out = match_spatial(x, ref)
        assert out.shape[-2:] == (15, 15)

    def test_handles_mismatched_height_and_width_independently(self):
        x = torch.randn(2, 4, 10, 20)
        ref = torch.randn(2, 4, 12, 18)
        out = match_spatial(x, ref)
        assert out.shape[-2:] == (12, 18)


class TestUNetDownBlock:
    def test_halves_spatial_dims_for_even_input(self):
        block = UNetDownBlock(3, 8)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 8, 16, 16)

    def test_normalize_false_still_produces_correct_shape(self):
        block = UNetDownBlock(3, 8, normalize=False)
        out = block(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 8, 16, 16)


class TestUNetUpBlock:
    def test_doubles_and_concatenates_skip(self):
        block = UNetUpBlock(16, 8)
        x = torch.randn(2, 16, 8, 8)
        skip = torch.randn(2, 8, 16, 16)
        out = block(x, skip)
        # out_channels (8) + skip's channels (8) = 16
        assert out.shape == (2, 16, 16, 16)

    def test_handles_odd_sized_skip_via_match_spatial(self):
        """The exact scenario match_spatial exists for: an odd-sized encoder feature map that a
        stride-2 transposed conv can't land on exactly by doubling an even bottleneck size."""
        block = UNetUpBlock(16, 8)
        x = torch.randn(2, 16, 7, 7)  # doubles to 14x14
        skip = torch.randn(2, 8, 15, 15)  # one pixel larger, from an odd-sized encoder input
        out = block(x, skip)
        assert out.shape == (2, 16, 15, 15)


class TestResidualBlock:
    def test_preserves_shape(self):
        block = ResidualBlock(32)
        x = torch.randn(2, 32, 28, 28)
        out = block(x)
        assert out.shape == x.shape

    def test_is_a_true_residual_connection(self):
        """Zeroing the block's conv weights should make it the identity function (x + 0)."""
        block = ResidualBlock(4)
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()
        x = torch.randn(1, 4, 10, 10)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-6)


class TestPatchGANDiscriminator:
    def test_output_is_a_patch_map_not_a_scalar(self):
        disc = PatchGANDiscriminator(in_channels=5)
        x = torch.randn(2, 5, 112, 112)
        out = disc(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 1
        assert out.shape[2] > 1 and out.shape[3] > 1

    def test_works_across_all_three_real_dataset_patch_sizes(self):
        """112 (SARptical), 120 (BigEarthNet), 256 (SEN1-2) -- see docs/RESEARCH_PLAN.md §4."""
        disc = PatchGANDiscriminator(in_channels=3)
        for size in (112, 120, 256):
            out = disc(torch.randn(1, 3, size, size))
            assert out.shape[0] == 1
            assert out.shape[1] == 1

    def test_concatenated_pix2pix_style_input_channels(self):
        """pix2pix conditions the discriminator on SAR+optical concatenated together."""
        sar_channels, optical_channels = 2, 3
        disc = PatchGANDiscriminator(in_channels=sar_channels + optical_channels)
        x = torch.randn(1, sar_channels + optical_channels, 120, 120)
        out = disc(x)
        assert out.shape[1] == 1
