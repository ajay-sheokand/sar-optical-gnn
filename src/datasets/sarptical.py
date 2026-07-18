"""
SARptical loader — a real, downloaded stretch dataset (docs/RESEARCH_PLAN.md §4's "other
datasets found during the review": "much higher resolution than Sentinel-1/2; a stretch option
for a 'does this hold at higher resolution' check").

Why this dataset got downloaded before BigEarthNet: BigEarthNet's real download is ~110GB
(measured directly, not estimated — see docs/BUILD_LOG.md), and the plan only needs a 20-50K
patch subset out of its ~549K total, with no built-in way to fetch a subset. SARptical is ~1GB,
directly downloadable with no login/form/registration wall (unlike QXS-SAROPT, which gates its
download behind a Chinese survey-form platform that would require submitting personal information
on your behalf — not something to automate), and has 10,108 real pairs — enough to prove the whole
data -> graph -> (eventually model) pipeline against real files today, before committing to
BigEarthNet's much larger download.

On-disk format, verified directly against the actual downloaded archive (not assumed from the
dataset's paper/README, which turned out to have a minor inaccuracy — see below):
    patch_SAR_OPT_SQUARE/
        point_{id}_ampPatch.mat              -- SAR amplitude patch, one per point
        point_{id}_{tile_ref}.tif.png         -- one or more optical patches per point
        ...
    selectedPntXYZ.mat                        -- point index + UTM coordinates (not used here)
    README.txt

Two things confirmed empirically rather than trusted from the README:
  1. The .mat file's internal variable name is 'ampCrop', not 'ampPatch' as the filename
     suggests -- `scipy.io.loadmat(...).keys()` was checked directly against a real extracted
     file. Using the wrong key name would have failed loudly (KeyError), not silently, but it's
     worth recording that the filename and the internal key don't match.
  2. There are 8,840 unique SAR points but 10,108 optical images -- some points have more than one
     matching optical patch (different viewing-angle tiles). Verified there are zero orphans in
     either direction (every optical file's point id has a matching .mat file, and vice versa) by
     scanning the full real archive's file listing before writing this loader, not assumed.

Design choice: this loader treats each (SAR, optical) *pair* as one sample, not each unique SAR
point -- so a point with 2 optical matches produces 2 dataset entries, both using the same SAR
array. This is what makes len(dataset) == 10,108, matching the dataset's own documented pair
count, rather than 8,840 (the unique-point count). The alternative (one entry per point, picking
only one optical image) would silently discard ~1,268 real, valid pairs.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
from PIL import Image

#: The .mat file's internal variable name, confirmed against a real extracted sample -- see the
#: module docstring's point 1. If a future re-download of this dataset uses a different variable
#: name, this will raise a clear KeyError rather than silently return the wrong array.
_SAR_MAT_KEY = "ampCrop"

_MAT_FILENAME_RE = re.compile(r"point_(\d+)_ampPatch\.mat$")
_PNG_FILENAME_RE = re.compile(r"point_(\d+)_.*\.png$")


def _find_pairs(root: str) -> list[tuple[str, str]]:
    """
    Scan `root` for SARptical SAR/optical files and return one (sar_path, optical_path) tuple per
    optical image found -- see the module docstring's design-choice note for why this can return
    more entries than there are unique SAR points.

    Split out from the dataset class for the same reason as src/datasets/sen1_2.py's
    `_find_pairs`: testable against cheap placeholder files, no real download needed for that part
    of the test suite.
    """
    mat_paths_by_point: dict[str, str] = {}
    png_paths_by_point: dict[str, list[str]] = {}

    for path in sorted(glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)):
        match = _MAT_FILENAME_RE.search(os.path.basename(path))
        if match:
            mat_paths_by_point[match.group(1)] = path

    for path in sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True)):
        match = _PNG_FILENAME_RE.search(os.path.basename(path))
        if match:
            png_paths_by_point.setdefault(match.group(1), []).append(path)

    mat_points = set(mat_paths_by_point)
    png_points = set(png_paths_by_point)
    unpaired = mat_points.symmetric_difference(png_points)
    if unpaired:
        raise ValueError(
            f"found {len(unpaired)} SARptical point(s) with a SAR patch but no optical match "
            f"(or vice versa) under {root!r} — e.g. {sorted(unpaired)[:5]}. This usually means an "
            f"incomplete extraction; re-check the source archive."
        )

    pairs = []
    for point_id in sorted(mat_points):
        sar_path = mat_paths_by_point[point_id]
        for optical_path in sorted(png_paths_by_point[point_id]):
            pairs.append((sar_path, optical_path))
    return pairs


def _read_sar_mat(path: str | Path) -> np.ndarray:
    """Read a SARptical .mat SAR patch and return it as an (H, W, 1) float32 array."""
    mat = scipy.io.loadmat(path)
    amplitude = mat[_SAR_MAT_KEY]
    return amplitude[:, :, np.newaxis].astype(np.float32)


def _read_optical_png(path: str | Path) -> np.ndarray:
    """Read a SARptical .png optical patch and return it as an (H, W, 3) float32 array."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.float32)


class SARpticalDataset:
    """
    Loader for the SARptical stretch dataset.

    Usage:
        ds = SARpticalDataset(root="data/sarptical/patch_SAR_OPT_SQUARE")
        sample = ds[0]
        sample["sar"].shape       # (112, 112, 1) -- SAR amplitude, dB scale
        sample["optical"].shape   # (112, 112, 3) -- RGB

    No `download=True` option, same reasoning as SEN1_2Dataset: this was a deliberate one-time
    manual download (see docs/BUILD_LOG.md), not something to automate on every dataset
    instantiation.
    """

    def __init__(self, root: str = "data/sarptical/patch_SAR_OPT_SQUARE") -> None:
        self.root = root
        self._pairs = _find_pairs(root)
        if not self._pairs:
            raise FileNotFoundError(
                f"no SARptical SAR/optical pairs found under {root!r} — has the dataset been "
                f"downloaded and extracted there yet?"
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sar_path, optical_path = self._pairs[index]
        return {
            "sar": _read_sar_mat(sar_path),
            "optical": _read_optical_png(optical_path),
            "sar_path": sar_path,
            "optical_path": optical_path,
        }
