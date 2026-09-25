"""Preprocessing: shapes, spacing, orientation and normalisation ranges."""

import json

import numpy as np
import pytest

from mri_jepa.data import preprocess as pp


def test_load_volume_is_channel_first_with_affine(phantom):
    vol = pp.load_volume(phantom["image"])
    assert tuple(vol.shape) == (1, 40, 48, 20)
    assert np.allclose(vol.pixdim[:3], [1.5, 1.25, 3.0])


def test_to_ras_isotropic_gives_1mm_ras(phantom):
    vol = pp.to_ras_isotropic(pp.load_volume(phantom["image"]))
    affine = vol.affine.numpy()
    # RAS with 1 mm voxels: the direction part of the affine is the identity.
    assert np.allclose(affine[:3, :3], np.eye(3), atol=1e-4)
    # 60 mm field of view along every axis -> about 60 voxels.
    assert all(abs(n - 60) <= 2 for n in vol.shape[1:])


def test_mask_lands_on_the_image_grid(phantom):
    image = pp.to_ras_isotropic(pp.load_volume(phantom["image"]))
    mask = pp.mask_on_grid(pp.load_volume(phantom["anat"]), image)
    assert mask.shape == tuple(image.shape[1:]) and mask.dtype == bool
    # The phantom's tissue is bright (>~250) and the air is ~0, so the mask
    # should sit on the bright voxels.
    img = image[0].numpy()
    assert img[mask].mean() > 400 and abs(img[~mask].mean()) < 60


def test_bounding_box():
    m = np.zeros((10, 10, 10), bool)
    m[2:5, 3:7, 4] = True
    assert pp.bounding_box(m) == (slice(2, 5), slice(3, 7), slice(4, 5))
    assert pp.bounding_box(m, margin=3) == (slice(0, 8), slice(0, 10), slice(1, 8))
    with pytest.raises(ValueError, match="empty"):
        pp.bounding_box(np.zeros((4, 4, 4), bool))


def test_normalize_intensity_range_and_background():
    rng = np.random.default_rng(0)
    image = rng.normal(100, 30, (30, 30, 30)).astype(np.float32)
    mask = np.zeros(image.shape, bool)
    mask[5:25, 5:25, 5:25] = True
    image[~mask] = 0.0  # air

    out, lo, hi = pp.normalize_intensity(image, mask)
    assert out.dtype == np.float32
    assert out.min() == -1.0 and out.max() == 1.0
    assert (out[~mask] == -1.0).all()
    # Percentiles come from the foreground, not the air: lo is ~ 100 - 2.33*30.
    assert 20 < lo < 50 and 150 < hi < 180
    # About 1% of foreground voxels are clipped at each end.
    fg = out[mask]
    assert 0.005 < (fg == 1.0).mean() < 0.02
    assert 0.005 < (fg == -1.0).mean() < 0.02


def test_normalize_intensity_rejects_constant_image():
    mask = np.ones((4, 4, 4), bool)
    with pytest.raises(ValueError, match="constant"):
        pp.normalize_intensity(np.full((4, 4, 4), 7.0), mask)


def test_preprocess_volume_end_to_end(phantom):
    arrays, info = pp.preprocess_volume(phantom["image"], phantom["anat"], phantom["anon"])
    image, anat, anon = arrays["image"], arrays["anat"], arrays["anon"]

    # Shapes: cropped to the ellipsoid (radii 22, 26, 24 mm) at 1 mm, stored (z, y, x).
    z, y, x = image.shape
    assert abs(z - 49) <= 3 and abs(y - 53) <= 3 and abs(x - 45) <= 3
    assert anat.shape == anon.shape == image.shape
    assert info["shape"] == [z, y, x] and info["axes"] == "ZYX"
    assert info["spacing_mm"] == [1.0, 1.0, 1.0]
    assert info["original_spacing_mm"] == [1.5, 1.25, 3.0]

    # Types and value ranges.
    assert image.dtype == np.float16 and anat.dtype == np.uint8
    assert set(np.unique(anat)) <= {0, 1}
    assert image.min() >= -1.0 and image.max() <= 1.0
    assert (image[anat == 0] == -1.0).all()
    # The crop is tight: the mask touches every face of the box.
    assert anat[0].any() and anat[-1].any() and anat[:, 0].any() and anat[:, :, -1].any()

    # Orientation: the phantom is brighter toward the top of the head, so the
    # mean of each axial slice should rise with z.
    means = [image[k][anat[k] == 1].astype(np.float32).mean() for k in range(z)]
    assert np.corrcoef(np.arange(z), means)[0, 1] > 0.9
    # The "face" (deface mask) is at the front (high y) and bottom (low z).
    zz, yy, _ = np.nonzero(anon)
    assert zz.mean() < z / 2 and yy.mean() > y / 2


def test_process_row_writes_files_and_resumes(raw_manifest, tmp_path):
    raw, manifest = raw_manifest
    out = tmp_path / "processed"
    row = manifest.iloc[0].to_dict()

    assert pp.process_row(row, raw, out, pp.PreprocessConfig()) == (row["id"], "ok", "")
    paths = pp.output_paths(out, row["dataset_id"], row["id"])
    info = json.loads(paths["json"].read_text())
    assert info["modality"] == "T1w" and info["source_dataset"] == "ds000000"
    assert info["files"] == {
        "image": paths["image"].name,
        "anat": paths["anat"].name,
        "anon": paths["anon"].name,
    }
    assert list(np.load(paths["image"]).shape) == info["shape"]
    assert not list(out.rglob("*.tmp"))  # no half-written leftovers

    # Second run: nothing to do. With overwrite: redone.
    assert pp.process_row(row, raw, out, pp.PreprocessConfig())[1] == "skipped"
    assert pp.process_row(row, raw, out, pp.PreprocessConfig(), overwrite=True)[1] == "ok"


def test_row_without_deface_mask(raw_manifest, tmp_path):
    raw, manifest = raw_manifest
    row = manifest[manifest["modality"] == "FLAIR"].iloc[0].to_dict()
    row["anon_mask"] = float("nan")  # what pandas gives for an empty CSV cell
    vid, status, msg = pp.process_row(row, raw, tmp_path, pp.PreprocessConfig())
    assert status == "ok", msg
    info = json.loads(pp.output_paths(tmp_path, row["dataset_id"], vid)["json"].read_text())
    assert info["files"]["anon"] is None


def test_bad_row_fails_without_crashing(raw_manifest, tmp_path):
    raw, manifest = raw_manifest
    row = manifest.iloc[0].to_dict() | {"image": "does/not/exist.nii.gz"}
    vid, status, msg = pp.process_row(row, raw, tmp_path, pp.PreprocessConfig())
    assert status == "failed" and "exist" in msg
    assert not pp.output_paths(tmp_path, row["dataset_id"], vid)["json"].exists()


def test_parallel_matches_sequential(raw_manifest, tmp_path):
    raw, manifest = raw_manifest
    rows = manifest.to_dict("records")
    seq = pp.process_many(rows, raw, tmp_path / "seq", workers=1, progress=False)
    par = pp.process_many(rows, raw, tmp_path / "par", workers=2, progress=False)
    assert sorted(s for _, s, _ in seq) == sorted(s for _, s, _ in par) == ["ok"] * 3
    for row in rows:
        a = pp.output_paths(tmp_path / "seq", row["dataset_id"], row["id"])["image"]
        b = pp.output_paths(tmp_path / "par", row["dataset_id"], row["id"])["image"]
        assert np.array_equal(np.load(a), np.load(b))
