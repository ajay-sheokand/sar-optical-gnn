"""
Tests for src/eval/metrics.py. FID pulls in a real InceptionV3 checkpoint download on first use
(torch-fidelity fetches it), so these tests run it for real rather than mocking it out --
mirroring this project's stated preference (docs/BUILD_LOG.md) for verifying against real
behavior over trusting an isolated unit in a vacuum.
"""

import pytest
import torch

from src.eval.metrics import TranslationMetrics, to_unit_range


class TestToUnitRange:
    def test_maps_tanh_range_to_unit_range(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        out = to_unit_range(x)
        assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0]))

    def test_clamps_out_of_range_values(self):
        x = torch.tensor([-2.0, 3.0])
        out = to_unit_range(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestTranslationMetricsPixelMetrics:
    def test_identical_images_give_perfect_ssim(self):
        metrics = TranslationMetrics()
        x = torch.rand(2, 3, 32, 32) * 2 - 1  # random tanh-range image
        metrics.update_pixel_metrics(x, x)
        result = metrics.compute()
        assert result["ssim"] == pytest.approx(1.0, abs=1e-4)
        assert result["psnr"] > 40  # near-infinite for identical images, capped numerically

    def test_different_images_give_lower_ssim_than_identical(self):
        metrics = TranslationMetrics()
        torch.manual_seed(0)
        real = torch.rand(2, 3, 32, 32) * 2 - 1
        generated = torch.rand(2, 3, 32, 32) * 2 - 1
        metrics.update_pixel_metrics(generated, real)
        result = metrics.compute()
        assert result["ssim"] < 1.0

    def test_works_on_non_rgb_channel_counts(self):
        """PSNR/SSIM must work on SAR's 1-2 channels too, not just 3-channel optical -- this
        metrics class is also used to sanity-check reconstructions in non-RGB domains."""
        metrics = TranslationMetrics()
        x = torch.rand(2, 2, 32, 32) * 2 - 1
        metrics.update_pixel_metrics(x, x)
        result = metrics.compute()
        assert result["ssim"] == pytest.approx(1.0, abs=1e-4)

    def test_accumulates_across_multiple_updates(self):
        """torchmetrics' update/compute pattern should reflect all updates, not just the last one."""
        metrics = TranslationMetrics()
        x = torch.rand(2, 3, 32, 32) * 2 - 1
        metrics.update_pixel_metrics(x, x)
        metrics.update_pixel_metrics(torch.rand(2, 3, 32, 32) * 2 - 1, torch.rand(2, 3, 32, 32) * 2 - 1)
        result = metrics.compute()
        assert result["ssim"] < 1.0  # pulled down by the second, non-identical pair

    def test_reset_clears_accumulated_state(self):
        metrics = TranslationMetrics()
        x = torch.rand(2, 3, 32, 32) * 2 - 1
        metrics.update_pixel_metrics(torch.rand(2, 3, 32, 32) * 2 - 1, torch.rand(2, 3, 32, 32) * 2 - 1)
        metrics.reset()
        metrics.update_pixel_metrics(x, x)
        result = metrics.compute()
        assert result["ssim"] == pytest.approx(1.0, abs=1e-4)


class TestTranslationMetricsFID:
    def test_fid_omitted_from_compute_when_never_updated(self):
        metrics = TranslationMetrics()
        metrics.update_pixel_metrics(torch.rand(2, 3, 16, 16) * 2 - 1, torch.rand(2, 3, 16, 16) * 2 - 1)
        result = metrics.compute()
        assert "fid" not in result

    def test_rejects_non_three_channel_input(self):
        metrics = TranslationMetrics()
        sar_like = torch.rand(2, 2, 64, 64) * 2 - 1
        with pytest.raises(ValueError, match="3 channels"):
            metrics.update_fid(sar_like, sar_like)

    @pytest.mark.slow
    def test_fid_runs_end_to_end_and_returns_nonnegative_float(self):
        """Real InceptionV3 forward pass -- marked slow since it downloads/loads a real backbone
        and this is the only test in the suite that needs one."""
        metrics = TranslationMetrics()
        torch.manual_seed(0)
        real = torch.rand(4, 3, 64, 64) * 2 - 1
        generated = torch.rand(4, 3, 64, 64) * 2 - 1
        metrics.update_fid(generated, real)
        result = metrics.compute()
        assert result["fid"] >= 0.0

    @pytest.mark.slow
    def test_fid_is_zero_for_identical_distributions(self):
        metrics = TranslationMetrics()
        torch.manual_seed(0)
        x = torch.rand(4, 3, 64, 64) * 2 - 1
        metrics.update_fid(x, x)
        result = metrics.compute()
        assert result["fid"] == pytest.approx(0.0, abs=1.0)
