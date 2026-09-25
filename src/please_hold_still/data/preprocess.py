"""Turn raw OpenMind NIfTI volumes into compact, normalised arrays.

For each volume:

1. Load image, anatomy mask and (optional) deface mask with nibabel.
   OpenMind stores the anatomy mask the other way round (1 = background), so
   it is flipped on load. Everything we write uses 1 = anatomy.
2. Reorient the image to RAS with MONAI, so array axes mean the same thing for
   every scan: x = left→right, y = posterior→anterior, z = inferior→superior.
3. Resample the image to 1 mm isotropic (trilinear). Put the masks onto exactly
   the same grid (nearest neighbour, so they stay 0/1).
4. Crop everything to the anatomy mask's bounding box.
5. Clip the image to its 1st–99th intensity percentiles, measured inside the
   anatomy mask only. Scale to [-1, 1] and set everything outside the mask
   to -1.
6. Save as .npy in (z, y, x) order, so one axial slice is one contiguous block
   and a few slices can be read without loading the whole file.
   image: float16; masks: uint8 (0/1). A JSON sidecar describes the result.

Output layout, one folder per OpenNeuro study:

    processed/<dataset_id>/<id>_image.npy
    processed/<dataset_id>/<id>_anat.npy
    processed/<dataset_id>/<id>_anon.npy     (only if a deface mask exists)
    processed/<dataset_id>/<id>.json         written last = "this volume is done"
"""

from __future__ import annotations

import json
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

with warnings.catch_warnings():  # MONAI's import triggers a torch.jit deprecation warning
    warnings.simplefilter("ignore", FutureWarning)
    from monai.data import MetaTensor
    from monai.transforms import Orientation, ResampleToMatch, Spacing

# Bump when the output changes, so re-running redoes older volumes.
# 2: OpenMind anatomy masks are flipped on load (they mark the background).
PREPROCESS_VERSION = 2
AXES = "ZYX"


@dataclass(frozen=True)
class PreprocessConfig:
    spacing_mm: float = 1.0
    lower_pct: float = 1.0
    upper_pct: float = 99.0
    margin: int = 0  # extra voxels kept around the anatomy bounding box


# ---------------------------------------------------------------------------
# Small building blocks (each one is unit-tested)
# ---------------------------------------------------------------------------


def load_volume(path: str | Path) -> MetaTensor:
    """Load a 3D NIfTI as a (1, X, Y, Z) float32 MetaTensor that carries its affine."""
    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    while data.ndim > 3 and data.shape[-1] == 1:  # (X, Y, Z, 1) -> (X, Y, Z)
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {data.shape} in {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"NaN or inf values in {path}")
    affine = torch.as_tensor(nii.affine, dtype=torch.float64)
    return MetaTensor(torch.from_numpy(np.ascontiguousarray(data))[None], affine=affine)


def load_anatomy_mask(path: str | Path) -> MetaTensor:
    """Load an OpenMind anatomy mask ("fb_mask") and flip it, so that 1 = anatomy.

    OpenMind stores 1 on the background and 0 on the head. The authors' loader
    (nnssl, `nnsslAnatDataLoader3D`) takes the foreground as `1 - anat`, and the
    real files agree: the head is where the stored mask is 0.
    """
    mask = load_volume(path)
    return MetaTensor((mask.as_tensor() < 0.5).float(), affine=mask.affine)


def to_ras_isotropic(image: MetaTensor, spacing_mm: float = 1.0) -> MetaTensor:
    """Reorient to RAS and resample to `spacing_mm` isotropic (trilinear)."""
    # labels: nibabel affines use the standard L→R, P→A, I→S world axes.
    image = Orientation(axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S")))(image)
    return Spacing(pixdim=spacing_mm, mode="bilinear", dtype=np.float32)(image)


def mask_on_grid(mask: MetaTensor, reference: MetaTensor) -> np.ndarray:
    """Resample a mask onto `reference`'s voxel grid (nearest neighbour). Returns bool (X, Y, Z).

    Works even if the mask was saved with a different orientation or resolution
    than the image. Voxels outside the mask's field of view become 0.
    """
    out = ResampleToMatch(mode="nearest", padding_mode="zeros", dtype=np.float32)(
        mask, img_dst=reference
    )
    return out[0].numpy() > 0.5


def bounding_box(mask: np.ndarray, margin: int = 0) -> tuple[slice, ...]:
    """Tightest box around the True voxels, grown by `margin` and clipped to the array."""
    if not mask.any():
        raise ValueError("anatomy mask is empty")
    box = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        nonzero = np.flatnonzero(mask.any(axis=other))
        lo = max(int(nonzero[0]) - margin, 0)
        hi = min(int(nonzero[-1]) + 1 + margin, mask.shape[axis])
        box.append(slice(lo, hi))
    return tuple(box)


def normalize_intensity(
    image: np.ndarray, mask: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0
) -> tuple[np.ndarray, float, float]:
    """Clip to percentiles measured inside `mask`, scale to [-1, 1], set outside to -1.

    Returns (normalised image as float32, low value, high value).
    """
    lo, hi = np.percentile(image[mask], [lower_pct, upper_pct])
    if not hi > lo:
        raise ValueError(f"image is constant inside the anatomy mask (value {lo})")
    out = (np.clip(image, lo, hi) - lo) / (hi - lo) * 2.0 - 1.0
    out[~mask] = -1.0
    return out.astype(np.float32), float(lo), float(hi)


def xyz_to_zyx(a: np.ndarray) -> np.ndarray:
    """(X, Y, Z) -> contiguous (Z, Y, X): axial slices become the first axis."""
    return np.ascontiguousarray(a.transpose(2, 1, 0))


# ---------------------------------------------------------------------------
# One volume, end to end
# ---------------------------------------------------------------------------


def preprocess_volume(
    image_path: str | Path,
    anat_mask_path: str | Path,
    anon_mask_path: str | Path | None = None,
    cfg: PreprocessConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run the whole pipeline on one volume. Returns (arrays, sidecar info)."""
    cfg = cfg or PreprocessConfig()
    raw = load_volume(image_path)
    original_shape = list(raw.shape[1:])
    original_spacing = [round(float(s), 4) for s in raw.pixdim[:3]]

    image = to_ras_isotropic(raw, cfg.spacing_mm)
    anat = mask_on_grid(load_anatomy_mask(anat_mask_path), image)
    anon = mask_on_grid(load_volume(anon_mask_path), image) if anon_mask_path else None

    # A mask the wrong way round would crop to the air and erase the head.
    # In MRI the head is brighter than the air around it, so check that.
    full = image[0].numpy()
    if (~anat).any() and not full[anat].mean() > full[~anat].mean():
        raise ValueError(
            "anatomy mask covers the darker part of the image (mean "
            f"{full[anat].mean():.3g} inside vs {full[~anat].mean():.3g} outside): "
            "is it the wrong way round?"
        )

    box = bounding_box(anat, cfg.margin)
    img = image[0].numpy()[box]
    anat = anat[box]
    normed, lo, hi = normalize_intensity(img, anat, cfg.lower_pct, cfg.upper_pct)

    # Affine of the cropped (x, y, z) grid: shift the origin to the box corner.
    shift = np.eye(4)
    shift[:3, 3] = [s.start for s in box]
    affine = image.affine.numpy() @ shift

    arrays = {
        "image": xyz_to_zyx(normed).astype(np.float16),
        "anat": xyz_to_zyx(anat).astype(np.uint8),
    }
    if anon is not None:
        arrays["anon"] = xyz_to_zyx(anon[box]).astype(np.uint8)

    info = {
        "shape": list(arrays["image"].shape),
        "axes": AXES,
        "orientation": "RAS",
        "spacing_mm": [cfg.spacing_mm] * 3,
        "dtype": "float16",
        "value_range": [-1.0, 1.0],
        "original_shape": original_shape,
        "original_spacing_mm": original_spacing,
        "crop_xyz": [[s.start, s.stop] for s in box],
        "affine_xyz": np.round(affine, 6).tolist(),
        "intensity_clip": {
            "lower_pct": cfg.lower_pct,
            "upper_pct": cfg.upper_pct,
            "lower_value": lo,
            "upper_value": hi,
        },
        "foreground_fraction": round(float(anat.mean()), 4),
        "preprocess_version": PREPROCESS_VERSION,
        "config": asdict(cfg),
    }
    return arrays, info


# ---------------------------------------------------------------------------
# Saving, resuming, running many volumes in parallel
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, write) -> None:
    """Write to `path.tmp`, then rename. A crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def output_paths(out_dir: Path, dataset_id: str, vid: str) -> dict[str, Path]:
    folder = Path(out_dir) / dataset_id
    return {
        "image": folder / f"{vid}_image.npy",
        "anat": folder / f"{vid}_anat.npy",
        "anon": folder / f"{vid}_anon.npy",
        "json": folder / f"{vid}.json",
    }


def _is_done(json_path: Path) -> bool:
    """True if the sidecar exists and was written by the current PREPROCESS_VERSION."""
    try:
        return json.loads(json_path.read_text()).get("preprocess_version") == PREPROCESS_VERSION
    except (OSError, ValueError):
        return False


def process_row(
    row: dict, raw_dir: Path, out_dir: Path, cfg: PreprocessConfig, overwrite: bool = False
) -> tuple[str, str, str]:
    """Preprocess one manifest row. Returns (id, "ok" | "skipped" | "failed", message)."""
    vid = row["id"]
    out = output_paths(out_dir, row["dataset_id"], vid)
    if not overwrite and _is_done(out["json"]):
        return vid, "skipped", ""
    try:
        anon = row.get("anon_mask")
        anon_path = raw_dir / anon if isinstance(anon, str) and anon else None
        arrays, info = preprocess_volume(
            raw_dir / row["image"], raw_dir / row["anat_mask"], anon_path, cfg
        )
        out["json"].parent.mkdir(parents=True, exist_ok=True)
        for key, arr in arrays.items():
            _atomic_write(out[key], lambda f, a=arr: np.save(f, a))
        info = {
            "id": vid,
            "modality": row["modality"],
            "source_dataset": row["dataset_id"],
            "source_image": row["image"],
            "files": {k: out[k].name if k in arrays else None for k in ("image", "anat", "anon")},
            **info,
        }
        _atomic_write(out["json"], lambda f: f.write(json.dumps(info, indent=2).encode()))
        return vid, "ok", ""
    except Exception as e:  # one bad scan must not stop the whole run
        return vid, "failed", f"{type(e).__name__}: {e}"


def _init_worker(threads: int) -> None:
    torch.set_num_threads(threads)


def process_many(
    rows: list[dict],
    raw_dir: Path,
    out_dir: Path,
    cfg: PreprocessConfig | None = None,
    workers: int = 1,
    overwrite: bool = False,
    progress: bool = True,
) -> list[tuple[str, str, str]]:
    """Preprocess many manifest rows, in parallel when `workers` > 1."""
    cfg = cfg or PreprocessConfig()
    from tqdm import tqdm

    results = []
    bar = tqdm(total=len(rows), desc="preprocess", disable=not progress)
    if workers <= 1:
        for row in rows:
            results.append(process_row(row, raw_dir, out_dir, cfg, overwrite))
            bar.update()
    else:
        # "spawn" starts clean worker processes; forking a process that already
        # runs torch threads can hang. Split the CPU threads between workers.
        threads = max(1, (os.cpu_count() or 1) // workers)
        with ProcessPoolExecutor(
            workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(threads,),
        ) as pool:
            futures = [
                pool.submit(process_row, row, raw_dir, out_dir, cfg, overwrite) for row in rows
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
                bar.update()
    bar.close()
    return results
