"""Shared test fixtures: small synthetic MRI volumes written as real NIfTI files."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest


def make_phantom(
    folder: Path,
    name: str,
    shape=(40, 48, 20),
    spacing=(1.5, 1.25, 3.0),
    radius_mm=(22.0, 26.0, 24.0),
    lps: bool = True,
    seed: int = 0,
) -> dict[str, Path]:
    """Write a fake head: an ellipsoid of tissue in air, plus its masks.

    - The image is brighter toward the top of the head (superior), so tests can
      check that the orientation is right after preprocessing.
    - Spacing is anisotropic, and with `lps=True` the axes are stored flipped
      (L-P-S instead of R-A-S), like many real scans.
    - The anatomy mask is stored the way OpenMind stores it: 1 on the air, 0 on
      the ellipsoid. The deface mask marks a "face" block at the
      anterior-inferior corner (1 = defaced).

    Returns the paths of the image, anatomy mask and deface mask.
    """
    rng = np.random.default_rng(seed)
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)
    center = (np.array(shape) - 1) / 2
    mm = (idx - center) * np.array(spacing)  # physical offset from the centre, in voxel axes
    inside = ((mm / np.array(radius_mm)) ** 2).sum(-1) <= 1.0

    sign = np.array([-1.0, -1.0, 1.0]) if lps else np.ones(3)
    superior = mm[..., 2] * sign[2]  # physical S coordinate
    image = np.where(inside, 500.0 + 10.0 * superior + rng.normal(0, 20, shape), 0.0)
    image += rng.normal(0, 2, shape)  # a little noise in the air too

    anterior = mm[..., 1] * sign[1]
    face = (anterior > 10) & (superior < -8)

    affine = np.diag([*(sign * np.array(spacing)), 1.0])
    affine[:3, 3] = -sign * center * np.array(spacing)  # centre of volume at world origin

    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, arr, dtype in (
        ("image", image, np.float32),
        ("anat", ~inside, np.uint8),
        ("anon", face, np.uint8),
    ):
        path = folder / f"{name}_{key}.nii.gz"
        nib.save(nib.Nifti1Image(arr.astype(dtype), affine), path)
        out[key] = path
    return out


@pytest.fixture
def phantom(tmp_path):
    return make_phantom(tmp_path / "raw", "sub-01_T1w")


@pytest.fixture
def raw_manifest(tmp_path):
    """A raw folder with 3 phantom volumes and the manifest the download step would write."""
    raw = tmp_path / "raw"
    rows = []
    for i, modality in enumerate(("T1w", "T2w", "FLAIR")):
        vid = f"ds00000{i}__sub-01__anat__sub-01_{modality}"
        files = make_phantom(raw / f"ds00000{i}", vid, seed=i, lps=i % 2 == 0)
        rows.append(
            {
                "id": vid,
                "dataset_id": f"ds00000{i}",
                "modality": modality,
                "image": str(files["image"].relative_to(raw)),
                "anat_mask": str(files["anat"].relative_to(raw)),
                # the FLAIR has no deface mask, like some real rows
                "anon_mask": str(files["anon"].relative_to(raw)) if modality != "FLAIR" else "",
                "image_quality_score": 1.0,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(raw / "subset_manifest.csv", index=False)
    return raw, manifest
