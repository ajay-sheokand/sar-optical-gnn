"""
Tests for src/datasets/sarptical.py.

Unlike the other dataset tests in this directory, this one *could* run against the real
downloaded dataset (data/sarptical/ — see docs/BUILD_LOG.md for why this dataset specifically was
downloaded when the others weren't: it's ~1GB and directly downloadable with no login/form wall,
unlike BigEarthNet's ~110GB or QXS-SAROPT's form-gated access). These tests still use small
synthetic fixtures rather than depending on that real download being present on whatever machine
runs the test suite, but tests/datasets/test_sarptical_real_data.py (a separate file) does
exercise the loader against the real thing directly.
"""

import numpy as np
import pytest
import scipy.io
from PIL import Image

from src.datasets.sarptical import SARpticalDataset, _find_pairs


class TestFindPairs:
    def test_pairs_one_optical_image_per_point(self, tmp_path):
        (tmp_path / "point_1_ampPatch.mat").touch()
        (tmp_path / "point_1_2014_RGB_20_013_0598.tif.png").touch()

        pairs = _find_pairs(str(tmp_path))

        assert len(pairs) == 1

    def test_a_point_with_multiple_optical_images_produces_multiple_pairs(self, tmp_path):
        # The real, verified behavior this loader is built around (module docstring's design
        # choice note): SARptical has 8,840 unique SAR points but 10,108 optical images, because
        # some points have more than one matching optical patch.
        (tmp_path / "point_1_ampPatch.mat").touch()
        (tmp_path / "point_1_2014_RGB_20_013_0598.tif.png").touch()
        (tmp_path / "point_1_2014_RGB_20_014_0665.tif.png").touch()

        pairs = _find_pairs(str(tmp_path))

        assert len(pairs) == 2
        assert all(sar_path.endswith("point_1_ampPatch.mat") for sar_path, _ in pairs)

    def test_raises_on_sar_with_no_optical_match(self, tmp_path):
        (tmp_path / "point_1_ampPatch.mat").touch()
        # no matching png for point 1

        with pytest.raises(ValueError, match="SAR patch but no optical match"):
            _find_pairs(str(tmp_path))

    def test_raises_on_optical_with_no_sar_match(self, tmp_path):
        (tmp_path / "point_1_2014_RGB_20_013_0598.tif.png").touch()
        # no matching mat for point 1

        with pytest.raises(ValueError):
            _find_pairs(str(tmp_path))


def _write_fake_sar_mat(path, height=8, width=8, fill_value=1.5):
    scipy.io.savemat(str(path), {"ampCrop": np.full((height, width), fill_value, dtype=np.float64)})


def _write_fake_optical_png(path, height=8, width=8, fill_value=100):
    image = Image.fromarray(np.full((height, width, 3), fill_value, dtype=np.uint8), mode="RGB")
    image.save(path)


class TestSARpticalDataset:
    def test_loads_real_mat_and_png_with_correct_shapes(self, tmp_path):
        _write_fake_sar_mat(tmp_path / "point_1_ampPatch.mat")
        _write_fake_optical_png(tmp_path / "point_1_2014_RGB_20_013_0598.tif.png")

        dataset = SARpticalDataset(root=str(tmp_path))
        sample = dataset[0]

        assert sample["sar"].shape == (8, 8, 1)
        assert sample["optical"].shape == (8, 8, 3)

    def test_sar_values_preserved_through_mat_loading(self, tmp_path):
        _write_fake_sar_mat(tmp_path / "point_1_ampPatch.mat", fill_value=-12.5)
        _write_fake_optical_png(tmp_path / "point_1_2014_RGB_20_013_0598.tif.png")

        sample = SARpticalDataset(root=str(tmp_path))[0]

        assert np.all(sample["sar"] == -12.5)

    def test_len_counts_pairs_not_unique_points(self, tmp_path):
        _write_fake_sar_mat(tmp_path / "point_1_ampPatch.mat")
        _write_fake_optical_png(tmp_path / "point_1_2014_RGB_20_013_0598.tif.png")
        _write_fake_optical_png(tmp_path / "point_1_2014_RGB_20_014_0665.tif.png")

        assert len(SARpticalDataset(root=str(tmp_path))) == 2

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SARpticalDataset(root=str(tmp_path))
