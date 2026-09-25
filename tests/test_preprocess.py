"""Preprocessing: shapes, spacing, orientation and normalisation ranges."""

import json

import nibabel as nib
import numpy as np
import pytest
from conftest import make_phantom

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
    mask = pp.mask_on_grid(pp.load_anatomy_mask(phantom["anat"]), image)
    assert mask.shape == tuple(image.shape[1:]) and mask.dtype == bool
    # The phantom's tissue is bright (>~250) and the air is ~0, so the flipped
    # mask should sit on the bright voxels.
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

    # Written by an older preprocessing version: redone without --overwrite.
    paths["json"].write_text(json.dumps(info | {"preprocess_version": 1}))
    assert pp.process_row(row, raw, out, pp.PreprocessConfig())[1] == "ok"
    assert json.loads(paths["json"].read_text())["preprocess_version"] == pp.PREPROCESS_VERSION


def test_row_without_deface_mask(raw_manifest, tmp_path):
    raw, manifest = raw_manifest
    row = manifest[manifest["modality"] == "FLAIR"].iloc[0].to_dict()
    row["anon_mask"] = float("nan")  # what pandas gives for an empty CSV cell
    vid, status, msg = pp.process_row(row, raw, tmp_path, pp.PreprocessConfig())
    assert status == "ok", msg
    info = json.loads(pp.output_paths(tmp_path, row["dataset_id"], vid)["json"].read_text())
    assert info["files"]["anon"] is None


def test_mask_the_wrong_way_round_is_rejected(tmp_path, phantom):
    """A mask stored as 1 = anatomy (not OpenMind's 1 = background) must fail loudly."""
    nii = nib.load(phantom["anat"])
    path = tmp_path / "flipped_anat.nii.gz"
    nib.save(nib.Nifti1Image(1 - np.asanyarray(nii.dataobj), nii.affine), path)
    with pytest.raises(ValueError, match="wrong way round"):
        pp.preprocess_volume(phantom["image"], path)


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


# ---------------------------------------------------------------------------
# Shapes across spacings, and edge cases seen in real data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spacing, lps",
    [
        ((1.0, 1.0, 1.0), False),
        ((0.5, 0.5, 0.8), True),  # high-res
        ((0.9, 0.9, 5.0), True),  # thick-slice clinical FLAIR
        ((2.0, 1.0, 1.0), False),
    ],
)
def test_output_shape_follows_physical_size(tmp_path, spacing, lps):
    """Same 44 x 52 x 48 mm ellipsoid, stored at different resolutions -> same 1 mm crop."""
    fov_mm = 64.0
    shape = tuple(int(round(fov_mm / s)) for s in spacing)
    f = make_phantom(tmp_path, "v", shape=shape, spacing=spacing, lps=lps)
    arrays, info = pp.preprocess_volume(f["image"], f["anat"])
    z, y, x = arrays["image"].shape
    # Ellipsoid diameters are 44 (x), 52 (y), 48 (z) mm; allow for voxelisation.
    assert abs(x - 44) <= 3 and abs(y - 52) <= 3 and abs(z - 48) <= 4
    assert info["original_spacing_mm"] == [round(float(s), 4) for s in spacing]


def test_target_spacing_2mm_halves_the_shape(phantom):
    one, _ = pp.preprocess_volume(phantom["image"], phantom["anat"])
    two, info = pp.preprocess_volume(
        phantom["image"], phantom["anat"], cfg=pp.PreprocessConfig(spacing_mm=2.0)
    )
    assert info["spacing_mm"] == [2.0, 2.0, 2.0]
    for a, b in zip(one["image"].shape, two["image"].shape, strict=True):
        assert abs(a / 2 - b) <= 1.5


def test_margin_grows_the_crop_and_keeps_background(phantom):
    tight, _ = pp.preprocess_volume(phantom["image"], phantom["anat"])
    loose, info = pp.preprocess_volume(
        phantom["image"], phantom["anat"], cfg=pp.PreprocessConfig(margin=3)
    )
    assert all(b == a + 6 for a, b in zip(tight["image"].shape, loose["image"].shape, strict=True))
    assert (loose["image"][loose["anat"] == 0] == -1).all()
    assert not loose["anat"][0].any()  # the margin slices hold no anatomy


def test_mask_on_a_different_grid_is_aligned(tmp_path):
    """Masks saved at another resolution/orientation must still land on the image.

    f: image and masks on an anisotropic, LPS-stored grid.
    g: the same physical head, masks stored at 1 mm in RAS.
    Uses the asymmetric "face" (deface) mask too: a symmetric ellipsoid would
    hide a left/right or front/back flip.
    """
    f = make_phantom(tmp_path / "a", "v", lps=True)
    g = make_phantom(tmp_path / "b", "v", shape=(60, 60, 60), spacing=(1, 1, 1), lps=False)
    image = pp.to_ras_isotropic(pp.load_volume(f["image"]))

    def iou(a, b):
        return (a & b).sum() / (a | b).sum()

    for key, load, min_iou in (("anat", pp.load_anatomy_mask, 0.9), ("anon", pp.load_volume, 0.8)):
        a = pp.mask_on_grid(load(f[key]), image)
        b = pp.mask_on_grid(load(g[key]), image)
        assert a.any() and b.any()
        assert iou(a, b) > min_iou, key  # only boundary voxels may differ


def test_trailing_singleton_dimension_is_accepted(tmp_path, phantom):
    nii = nib.load(phantom["image"])
    path = tmp_path / "4d.nii.gz"
    nib.save(nib.Nifti1Image(np.asanyarray(nii.dataobj)[..., None], nii.affine), path)
    assert tuple(pp.load_volume(path).shape) == (1, 40, 48, 20)


def test_true_4d_volume_is_rejected(tmp_path):
    path = tmp_path / "4d.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8, 3), np.float32), np.eye(4)), path)
    with pytest.raises(ValueError, match="3D"):
        pp.load_volume(path)


def test_nan_in_image_is_rejected(tmp_path):
    data = np.ones((8, 8, 8), np.float32)
    data[2, 2, 2] = np.nan
    path = tmp_path / "nan.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    with pytest.raises(ValueError, match="NaN"):
        pp.load_volume(path)


@pytest.mark.parametrize("scale, offset", [(1.0, 0.0), (1000.0, 0.0), (0.01, -5.0), (3.0, 250.0)])
def test_normalization_ignores_scanner_units(scale, offset):
    """Scanners use arbitrary intensity units; the result must not depend on them."""
    rng = np.random.default_rng(1)
    image = rng.gamma(2.0, 50.0, (24, 24, 24)).astype(np.float64)
    mask = np.zeros(image.shape, bool)
    mask[4:20, 4:20, 4:20] = True
    ref, _, _ = pp.normalize_intensity(image, mask)
    out, _, _ = pp.normalize_intensity(image * scale + offset, mask)
    assert np.allclose(out, ref, atol=1e-5)
    assert out.min() == -1.0 and out.max() == 1.0


@pytest.mark.parametrize("lower, upper", [(0.0, 100.0), (0.5, 99.5), (5.0, 95.0)])
def test_normalization_range_for_other_percentiles(lower, upper):
    rng = np.random.default_rng(2)
    image = rng.normal(0, 1, (20, 20, 20))
    mask = np.ones(image.shape, bool)
    out, _, _ = pp.normalize_intensity(image, mask, lower, upper)
    assert out.min() == -1.0 and out.max() == 1.0
    clipped_top = (out == 1.0).mean()
    assert clipped_top == pytest.approx((100 - upper) / 100, abs=0.01)


def test_float16_storage_stays_within_range(phantom):
    arrays, _ = pp.preprocess_volume(phantom["image"], phantom["anat"])
    img = arrays["image"]
    assert img.dtype == np.float16
    assert float(img.min()) == -1.0 and float(img.max()) == 1.0
    assert np.isfinite(img.astype(np.float32)).all()


def test_preprocessing_is_deterministic(phantom):
    a, _ = pp.preprocess_volume(phantom["image"], phantom["anat"], phantom["anon"])
    b, _ = pp.preprocess_volume(phantom["image"], phantom["anat"], phantom["anon"])
    assert all(np.array_equal(a[k], b[k]) for k in a)
