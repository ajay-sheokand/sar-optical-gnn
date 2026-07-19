"""
Evaluation metrics for SAR-to-optical translation quality: PSNR, SSIM, FID.

Why these three and not more yet: docs/RESEARCH_PLAN.md §4 pins the SEN1-2 validation pass to
exactly these ("pix2pix PSNR ~28.0 / SSIM ~0.20-0.30 ... FID typically 90-120") -- landing near
those numbers on our own pix2pix/CycleGAN reimplementations is what certifies the data/training
pipeline before the novel model (M4) gets built on top of it. LPIPS is deferred to M4 (§8), since
it's used there as a training loss, not needed as an M3 baseline metric.

FID moved up from its originally-planned M5 slot to M3 specifically because the SEN1-2 validation
pass needs it now, not later -- see requirements.txt's M3 section for the torch-fidelity
dependency this pulled forward.

All three metrics here assume generator output in tanh's [-1, 1] range (the convention every
generator in src/models/ uses) -- PSNR/SSIM take that range directly via `data_range=2.0`; FID
needs its own [0, 1]-range copy (`to_unit_range`) since torchvision's InceptionV3 backbone expects
that regardless of what range the generator itself trained in.
"""

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance


def to_unit_range(x):
    """Map a tanh-range ([-1, 1]) tensor to [0, 1], the range FID's InceptionV3 backbone expects."""
    return (x.clamp(-1, 1) + 1) / 2


class TranslationMetrics:
    """
    Accumulates PSNR/SSIM (any channel count) and FID (3-channel RGB only -- see update_fid) over
    a full evaluation pass, matching torchmetrics' update()/compute() accumulation pattern so
    metrics are correct across multiple batches, not just averaged per-batch (which would bias
    FID in particular, since its Frechet distance is not a simple per-batch average).
    """

    def __init__(self, data_range=2.0, device="cpu"):
        self.psnr = PeakSignalNoiseRatio(data_range=data_range).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self._fid_updated = False

    def update_pixel_metrics(self, generated, real):
        """generated, real: (B, C, H, W) tensors in [-1, 1], any matching channel count."""
        self.psnr.update(generated, real)
        self.ssim.update(generated, real)

    def update_fid(self, generated_rgb, real_rgb):
        """
        generated_rgb, real_rgb: (B, 3, H, W) tensors in [-1, 1]. Raises if given anything other
        than 3 channels -- FID's InceptionV3 backbone is trained on RGB and has no defined
        behavior for SAR's 1-2 channels or optical's full multispectral stack, so this is a real
        constraint worth failing loudly on rather than silently feeding it the wrong shape (per
        docs/RESEARCH_PLAN.md §5's own caveat: "report cautiously, RGB-bands-only"). Callers with
        multispectral optical must select/compose an RGB triplet themselves before calling this.
        """
        for name, tensor in (("generated_rgb", generated_rgb), ("real_rgb", real_rgb)):
            if tensor.shape[1] != 3:
                raise ValueError(f"{name} must have exactly 3 channels for FID, got shape {tuple(tensor.shape)}")

        self.fid.update(to_unit_range(real_rgb), real=True)
        self.fid.update(to_unit_range(generated_rgb), real=False)
        self._fid_updated = True

    def compute(self):
        """Returns a dict of scalar floats. 'fid' is omitted if update_fid was never called."""
        result = {
            "psnr": self.psnr.compute().item(),
            "ssim": self.ssim.compute().item(),
        }
        if self._fid_updated:
            result["fid"] = self.fid.compute().item()
        return result

    def reset(self):
        self.psnr.reset()
        self.ssim.reset()
        self.fid.reset()
        self._fid_updated = False
