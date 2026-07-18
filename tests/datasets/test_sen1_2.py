"""
Tests for src/datasets/sen1_2.py — the one loader in this project that isn't backed by torchgeo,
so its file-pairing logic is entirely our own code and needs its own real test coverage (not just
"trust the upstream library").

Two levels of test, deliberately separated:
  1. `_find_pairs` / `_pair_key` tested against cheap, empty placeholder files (`tmp_path`,
     `Path.touch()`) — fast, and sufficient for testing whether *filenames* get matched correctly.
  2. `SEN1_2Dataset.__getitem__` tested against real tiny GeoTIFFs written with rasterio — slower,
     but validates the actual pixel-reading + channel-order-conversion path end to end.

Neither needs the real, multi-GB SEN1-2 download.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.datasets.sen1_2 import SEN1_2Dataset, _find_pairs, _pair_key


class TestPairKey:
    def test_matches_s1_and_s2_filenames_to_the_same_key(self):
        assert _pair_key("ROIs1970_fall_s1_p407.tif") == _pair_key("ROIs1970_fall_s2_p407.tif")

    def test_different_patches_get_different_keys(self):
        assert _pair_key("ROIs1970_fall_s1_p407.tif") != _pair_key("ROIs1970_fall_s1_p408.tif")

    def test_non_patch_filename_returns_none(self):
        # A README or metadata file sitting alongside the real patch files shouldn't be treated
        # as an orphaned SAR/optical file and shouldn't blow up _find_pairs.
        assert _pair_key("README.md") is None


class TestFindPairs:
    def test_pairs_matching_files(self, tmp_path):
        (tmp_path / "ROIs1970_fall_s1_p1.tif").touch()
        (tmp_path / "ROIs1970_fall_s2_p1.tif").touch()
        (tmp_path / "ROIs1970_fall_s1_p2.tif").touch()
        (tmp_path / "ROIs1970_fall_s2_p2.tif").touch()

        pairs = _find_pairs(str(tmp_path))

        assert len(pairs) == 2
        for sar_path, optical_path in pairs:
            assert "_s1_" in sar_path
            assert "_s2_" in optical_path

    def test_finds_pairs_in_nested_scene_folders(self, tmp_path):
        # Real SEN1-2 downloads split SAR/optical into separate per-scene subfolders, e.g.
        # ROIs1970_fall_s1/ and ROIs1970_fall_s2/ -- not flat alongside each other.
        s1_dir = tmp_path / "ROIs1970_fall_s1"
        s2_dir = tmp_path / "ROIs1970_fall_s2"
        s1_dir.mkdir()
        s2_dir.mkdir()
        (s1_dir / "ROIs1970_fall_s1_p1.tif").touch()
        (s2_dir / "ROIs1970_fall_s2_p1.tif").touch()

        pairs = _find_pairs(str(tmp_path))
        assert len(pairs) == 1

    def test_raises_on_unpaired_file(self, tmp_path):
        (tmp_path / "ROIs1970_fall_s1_p1.tif").touch()
        (tmp_path / "ROIs1970_fall_s2_p1.tif").touch()
        (tmp_path / "ROIs1970_fall_s1_p2.tif").touch()  # no matching s2 file for patch 2

        with pytest.raises(ValueError, match="no matching"):
            _find_pairs(str(tmp_path))


def _write_fake_geotiff(path, num_bands, height=8, width=8, fill_value=0):
    """Write a tiny, valid GeoTIFF so rasterio has something real to read in the test below."""
    # Arbitrary but non-trivial origin/pixel-size (rather than from_origin(0, 0, 1, 1)) so
    # rasterio doesn't warn about an identity-equivalent affine transform on every test run.
    transform = from_origin(500000, 4649200, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=num_bands,
        dtype="float32",
        transform=transform,
    ) as dst:
        for band in range(1, num_bands + 1):
            dst.write(np.full((height, width), fill_value + band, dtype=np.float32), band)


class TestSEN1_2Dataset:
    def test_loads_real_geotiff_pair_with_correct_shapes(self, tmp_path):
        _write_fake_geotiff(tmp_path / "ROIs1970_fall_s1_p1.tif", num_bands=1)
        _write_fake_geotiff(tmp_path / "ROIs1970_fall_s2_p1.tif", num_bands=3)

        dataset = SEN1_2Dataset(root=str(tmp_path))
        sample = dataset[0]

        assert sample["sar"].shape == (8, 8, 1)
        assert sample["optical"].shape == (8, 8, 3)
        # band 1 of the SAR file was filled with value (0 + 1) = 1 everywhere
        assert np.all(sample["sar"] == 1.0)

    def test_len_matches_number_of_pairs(self, tmp_path):
        for i in range(3):
            _write_fake_geotiff(tmp_path / f"ROIs1970_fall_s1_p{i}.tif", num_bands=1)
            _write_fake_geotiff(tmp_path / f"ROIs1970_fall_s2_p{i}.tif", num_bands=3)

        assert len(SEN1_2Dataset(root=str(tmp_path))) == 3

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SEN1_2Dataset(root=str(tmp_path))
