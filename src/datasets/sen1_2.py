"""
SEN1-2 loader — this project's validation harness (docs/RESEARCH_PLAN.md §4).

Unlike BigEarthNet and SEN12MS, torchgeo has no built-in dataset class for SEN1-2, so this module
parses the raw file layout directly using `rasterio` (already a project dependency).

Why SEN1-2 has a different job from the other two datasets: it has no land-cover labels at all, so
it can't be used for the downstream classification check or for training the final model. Its job
is narrower and more specific — it's the dataset the literature review found published PSNR/SSIM/
FID numbers for pix2pix and CycleGAN on (see docs/LITERATURE_REVIEW.md and docs/RESEARCH_PLAN.md
§4), so reproducing those two baselines here and checking the numbers land in the right
neighborhood is how M3 validates the training/metrics pipeline *before* trusting it on the real
experiment (BigEarthNet). Same instinct as checking a new regression implementation against a
textbook dataset with known coefficients before trusting it on real research data
(docs/UNDERSTANDING_THE_PROJECT.md §6).

On-disk layout assumed here, per the dataset's documented naming convention (Schmitt et al. 2018 —
same research group and file-naming style as SEN12MS): SAR and optical patches are distributed in
separate per-scene folders (e.g. `ROIs1970_fall_s1/`, `ROIs1970_fall_s2/`), each containing patch
files whose names differ only in the "_s1_" / "_s2_" marker (e.g. `ROIs1970_fall_s1_p407.tif` pairs
with `ROIs1970_fall_s2_p407.tif`). This has NOT been verified against an actual downloaded copy of
the dataset (that download is a separate, deliberate multi-GB step — see the module-level note
below) — if the real folder layout differs in some way this docstring didn't anticipate, the fix is
localized to `_find_pairs()` below, which is the one place that encodes this assumption.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from src.datasets.common import chw_to_hwc

_S1_MARKER = "_s1_"
_S2_MARKER = "_s2_"


def _pair_key(filename: str) -> str | None:
    """
    Derive a key shared by a SAR/optical patch pair from a filename, e.g.
    "ROIs1970_fall_s1_p407.tif" and "ROIs1970_fall_s2_p407.tif" both map to
    "ROIs1970_fall_p407.tif". Returns None if the filename contains neither marker (not a
    SEN1-2 patch file — ignored rather than treated as an error, since real download trees often
    include README/metadata files alongside the patches).
    """
    if _S1_MARKER in filename:
        return filename.replace(_S1_MARKER, "_")
    if _S2_MARKER in filename:
        return filename.replace(_S2_MARKER, "_")
    return None


def _find_pairs(root: str) -> list[tuple[str, str]]:
    """
    Recursively scan `root` for SEN1-2 patch files and pair up SAR/optical files that share a
    patch identity. Returns a sorted list of (sar_path, optical_path) tuples.

    Split out from the dataset class specifically so the *pairing logic* can be unit-tested with
    cheap, empty placeholder files (no real GeoTIFF content needed to test whether filenames get
    paired correctly) — see tests/datasets/test_sen1_2.py.

    Raises:
        ValueError: if any patch key has an SAR file but no matching optical file (or vice versa)
            — a silently-dropped unpaired file would be a correctness bug for a task that needs
            paired training data, so this fails loudly instead of skipping it quietly.
    """
    sar_files: dict[str, str] = {}
    optical_files: dict[str, str] = {}

    for path in sorted(glob.glob(os.path.join(root, "**", "*.tif"), recursive=True)):
        filename = os.path.basename(path)
        key = _pair_key(filename)
        if key is None:
            continue
        if _S1_MARKER in filename:
            sar_files[key] = path
        else:
            optical_files[key] = path

    sar_keys = set(sar_files)
    optical_keys = set(optical_files)
    unpaired = sar_keys.symmetric_difference(optical_keys)
    if unpaired:
        raise ValueError(
            f"found {len(unpaired)} SEN1-2 patch file(s) with no matching SAR/optical "
            f"counterpart under {root!r} — e.g. {sorted(unpaired)[:5]}. This usually means an "
            f"incomplete download; re-check the source archive."
        )

    return [(sar_files[key], optical_files[key]) for key in sorted(sar_keys)]


class SEN1_2Dataset:
    """
    Loader for the SEN1-2 paired SAR/optical benchmark dataset.

    Usage:
        ds = SEN1_2Dataset(root="data/sen1_2")
        sample = ds[0]
        sample["sar"].shape       # (256, 256, 1) -- single-band VV backscatter
        sample["optical"].shape   # (256, 256, 3) -- Sentinel-2 RGB composite

    No `download=True` option: SEN1-2 is distributed via mediaTUM
    (https://mediaTUM.ub.tum.de/1436631), which requires manually accepting the dataset's terms —
    not something to automate. Download it yourself first, then point `root` at the extracted
    directory.
    """

    def __init__(self, root: str = "data/sen1_2") -> None:
        self.root = root
        self._pairs = _find_pairs(root)
        if not self._pairs:
            raise FileNotFoundError(
                f"no SEN1-2 SAR/optical patch pairs found under {root!r} — has the dataset been "
                f"downloaded and extracted there yet?"
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sar_path, optical_path = self._pairs[index]
        return {
            "sar": _read_geotiff_hwc(sar_path),
            "optical": _read_geotiff_hwc(optical_path),
            "sar_path": sar_path,
            "optical_path": optical_path,
        }


def _read_geotiff_hwc(path: str | Path) -> np.ndarray:
    """Read a GeoTIFF via rasterio and return it as a (H, W, C) float32 array."""
    with rasterio.open(path) as dataset:
        array = dataset.read()  # rasterio returns (C, H, W), matching torchgeo's convention
    return chw_to_hwc(array)
