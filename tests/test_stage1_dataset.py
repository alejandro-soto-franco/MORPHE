"""Regression tests for the stray-file and train/val-split bugs (see README)."""

from PIL import Image

from morphe.datasets.stage1_dataset import Stage1Dataset, list_image_files, region_names, split_by_region


def _make_region_images(tmp_path, names):
    for name in names:
        Image.new("RGB", (16, 16), color=(255, 0, 0)).save(tmp_path / f"{name}.png")


def test_list_image_files_ignores_stray_non_image_files(tmp_path):
    _make_region_images(tmp_path, ["region_0", "region_1"])
    (tmp_path / "1").write_text("1")
    (tmp_path / "111").write_text("111")

    files = list_image_files(tmp_path)

    assert {p.stem for p in files} == {"region_0", "region_1"}


def test_stage1_dataset_ignores_stray_files(tmp_path):
    _make_region_images(tmp_path, ["region_0", "region_1"])
    (tmp_path / "1").write_text("1")

    ds = Stage1Dataset(tmp_path, masks_per_image=2)

    assert len(ds) == 2 * 2
    masked_img, img, bbox = ds[0]
    assert masked_img.shape == img.shape
    assert bbox.shape == (4,)


def test_split_by_region_holds_out_explicit_regions(tmp_path):
    _make_region_images(tmp_path, [f"region_{i}" for i in range(4)])

    train_ds, val_ds = split_by_region(tmp_path, val_regions=("region_3",))

    assert set(region_names(tmp_path)) == {"region_0", "region_1", "region_2", "region_3"}
    assert {p.stem for p in train_ds.img_files} == {"region_0", "region_1", "region_2"}
    assert {p.stem for p in val_ds.img_files} == {"region_3"}


def test_split_by_region_rejects_unknown_val_region(tmp_path):
    import pytest

    _make_region_images(tmp_path, ["region_0"])
    with pytest.raises(ValueError):
        split_by_region(tmp_path, val_regions=("region_99",))
