"""
Tests for src/datasets/sen1_2.py — the one loader in this project that isn't backed by torchgeo,
so its file-pairing logic is entirely our own code and needs its own real test coverage (not just
"trust the upstream library").

Two levels of test, deliberately separated:
  1. `_find_pairs` / `_pair_key` tested against cheap, empty placeholder files (`tmp_path`,
     `Path.touch()`) — fast, and sufficient for testing whether *filenames* get matched correctly.
  2. `SEN1_2Dataset.__getitem__` tested against real tiny PNGs written with rasterio — slower,
     but validates the actual pixel-reading + channel-order-conversion path end to end.

Neither needs the real, multi-GB SEN1-2 download -- though the file layout assumed here (per-scene
`s1_<scene>/`, `s2_<scene>/` folders, `.png` files) has since been verified directly against a real
downloaded copy (docs/BUILD_LOG.md's M3 entry).
"""

import numpy as np
import pytest
import rasterio

from src.datasets.sen1_2 import SEN1_2Dataset, _find_pairs, _pair_key


class TestPairKey:
    def test_matches_s1_and_s2_filenames_to_the_same_key(self):
        assert _pair_key("ROIs1970_fall_s1_p407.png") == _pair_key("ROIs1970_fall_s2_p407.png")

    def test_different_patches_get_different_keys(self):
        assert _pair_key("ROIs1970_fall_s1_p407.png") != _pair_key("ROIs1970_fall_s1_p408.png")

    def test_non_patch_filename_returns_none(self):
        # A README or metadata file sitting alongside the real patch files shouldn't be treated
        # as an orphaned SAR/optical file and shouldn't blow up _find_pairs.
        assert _pair_key("README.md") is None


class TestFindPairs:
    def test_pairs_matching_files(self, tmp_path):
        (tmp_path / "ROIs1970_fall_s1_p1.png").touch()
        (tmp_path / "ROIs1970_fall_s2_p1.png").touch()
        (tmp_path / "ROIs1970_fall_s1_p2.png").touch()
        (tmp_path / "ROIs1970_fall_s2_p2.png").touch()

        pairs = _find_pairs(str(tmp_path))

        assert len(pairs) == 2
        for sar_path, optical_path in pairs:
            assert "_s1_" in sar_path
            assert "_s2_" in optical_path

    def test_finds_pairs_in_nested_scene_folders(self, tmp_path):
        # Real SEN1-2 downloads split SAR/optical into separate per-*scene* subfolders, e.g.
        # s1_0/ and s2_0/ (verified directly -- docs/BUILD_LOG.md's M3 entry) -- not flat
        # alongside each other, and not one folder per ROI as originally assumed.
        s1_dir = tmp_path / "s1_0"
        s2_dir = tmp_path / "s2_0"
        s1_dir.mkdir()
        s2_dir.mkdir()
        (s1_dir / "ROIs1970_fall_s1_0_p1.png").touch()
        (s2_dir / "ROIs1970_fall_s2_0_p1.png").touch()

        pairs = _find_pairs(str(tmp_path))
        assert len(pairs) == 1

    def test_raises_on_unpaired_file(self, tmp_path):
        (tmp_path / "ROIs1970_fall_s1_p1.png").touch()
        (tmp_path / "ROIs1970_fall_s2_p1.png").touch()
        (tmp_path / "ROIs1970_fall_s1_p2.png").touch()  # no matching s2 file for patch 2

        with pytest.raises(ValueError, match="no matching"):
            _find_pairs(str(tmp_path))


def _write_fake_png(path, num_bands, height=8, width=8, fill_value=0):
    """Write a tiny, valid PNG so rasterio has something real to read in the test below --
    matching the real dataset's actual format (uint8, no georeferencing), not the GeoTIFF format
    originally (and wrongly) assumed before a real download existed to check against."""
    with rasterio.open(
        path, "w", driver="PNG", height=height, width=width, count=num_bands, dtype="uint8"
    ) as dst:
        for band in range(1, num_bands + 1):
            dst.write(np.full((height, width), fill_value + band, dtype=np.uint8), band)


class TestSEN1_2Dataset:
    def test_loads_real_png_pair_with_correct_shapes(self, tmp_path):
        _write_fake_png(tmp_path / "ROIs1970_fall_s1_p1.png", num_bands=1)
        _write_fake_png(tmp_path / "ROIs1970_fall_s2_p1.png", num_bands=3)

        dataset = SEN1_2Dataset(root=str(tmp_path))
        sample = dataset[0]

        assert sample["sar"].shape == (8, 8, 1)
        assert sample["optical"].shape == (8, 8, 3)
        # band 1 of the SAR file was filled with value (0 + 1) = 1 everywhere
        assert np.all(sample["sar"] == 1.0)

    def test_len_matches_number_of_pairs(self, tmp_path):
        for i in range(3):
            _write_fake_png(tmp_path / f"ROIs1970_fall_s1_p{i}.png", num_bands=1)
            _write_fake_png(tmp_path / f"ROIs1970_fall_s2_p{i}.png", num_bands=3)

        assert len(SEN1_2Dataset(root=str(tmp_path))) == 3

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SEN1_2Dataset(root=str(tmp_path))
