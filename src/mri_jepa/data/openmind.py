"""OpenMind dataset: metadata, subset selection and download.

OpenMind (https://huggingface.co/datasets/AnonRes/OpenMind) has about 114k 3D
head-and-neck MRIs from 800 OpenNeuro studies. The Hugging Face repo holds:

    openneuro_metadata.csv          one row per image
    OpenMind/<image_path>           the NIfTI volumes (.nii.gz)
    OpenMind/<anat_mask_path>       anatomy masks: where there is tissue
    OpenMind/<anon_mask_path>       deface masks: regions removed for anonymisation

The column names come from the authors' own loader (MIC-DKFZ/nnssl,
dataset_conversion/Dataset001_OpenMind.py). The first part of `image_path` is
the OpenNeuro study id (e.g. "ds000117").

huggingface_hub is imported inside the functions that need it, so scripts can
call `configure_hf_cache()` first. The library reads its cache settings when
it is imported.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from mri_jepa import paths

REPO_ID = "AnonRes/OpenMind"
METADATA_FILENAME = "openneuro_metadata.csv"
DEFAULT_PREFIX = "OpenMind"
DEFAULT_MODALITIES = ("T1w", "T2w", "FLAIR")
REQUIRED_COLUMNS = ("image_path", "modality", "anat_mask_path")

# Columns of the manifest that the download step writes and preprocessing reads.
# `image`, `anat_mask`, `anon_mask` are paths relative to the raw data folder.
MANIFEST_NAME = "subset_manifest.csv"
MANIFEST_COLUMNS = [
    "id",
    "dataset_id",
    "modality",
    "image",
    "anat_mask",
    "anon_mask",
    "image_quality_score",
]


# ---------------------------------------------------------------------------
# Metadata and subset selection (pure pandas, no network)
# ---------------------------------------------------------------------------


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Read openneuro_metadata.csv and add a `dataset_id` column."""
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} is missing columns {missing}. Found: {list(df.columns)}")
    df["dataset_id"] = df["image_path"].str.split("/").str[0]
    return df


def volume_id(image_path: str) -> str:
    """A readable, unique, filename-safe id from an image's relative path.

    "ds000117/sub-01/ses-mri/anat/sub-01_T1w.nii.gz"
      -> "ds000117__sub-01__ses-mri__anat__sub-01_T1w"
    """
    stem = image_path
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem.replace("/", "__")


def _has_path(col: pd.Series) -> pd.Series:
    return col.notna() & (col.astype(str).str.strip() != "")


def select_subset(
    meta: pd.DataFrame,
    n_total: int,
    modalities: Iterable[str] = DEFAULT_MODALITIES,
    seed: int = 0,
    max_iqs: float | None = None,
) -> pd.DataFrame:
    """Pick about `n_total` images, split evenly across `modalities`.

    Within each modality we go round-robin over studies: one image from every
    study (in a seeded random order), then a second one from every study that
    has one, and so on. This spreads the subset over as many scanners and sites
    as possible.

    The order is fixed by `seed`, so a larger `n_total` picks a superset of a
    smaller one. Scaling up later reuses what is already downloaded.

    Images without an anatomy mask are skipped (preprocessing needs it). With
    `max_iqs`, images whose image_quality_score is above it are skipped (lower
    is better; the OpenMind authors use cutoffs between 1.5 and 3.5).
    """
    modalities = list(modalities)
    df = meta[meta["modality"].isin(modalities) & _has_path(meta["anat_mask_path"])]
    if max_iqs is not None:
        if "image_quality_score" not in df.columns:
            raise KeyError("--max-iqs needs an image_quality_score column")
        df = df[df["image_quality_score"] <= max_iqs]

    base, extra = divmod(n_total, len(modalities))
    picked = []
    for i, modality in enumerate(modalities):
        k = base + (1 if i < extra else 0)
        rows = df[df["modality"] == modality].sample(frac=1.0, random_state=seed)
        rank = rows.groupby("dataset_id").cumcount()
        # A stable sort keeps the shuffled order among images with equal rank.
        rows = rows.assign(_rank=rank.values).sort_values("_rank", kind="stable")
        if len(rows) < k:
            print(f"warning: only {len(rows)} {modality} images available, wanted {k}")
        picked.append(rows.head(k).drop(columns="_rank"))
    return pd.concat(picked, ignore_index=True)


def repo_path(relative_path: str, prefix: str = DEFAULT_PREFIX) -> str:
    """Path of a CSV-relative file inside the Hugging Face repo."""
    return f"{prefix}/{relative_path}" if prefix else relative_path


def build_manifest(subset: pd.DataFrame, prefix: str = DEFAULT_PREFIX) -> pd.DataFrame:
    """Turn selected metadata rows into the manifest that preprocessing reads."""
    anon = subset.get("anon_mask_path", pd.Series("", index=subset.index))
    manifest = pd.DataFrame(
        {
            "id": subset["image_path"].map(volume_id),
            "dataset_id": subset["dataset_id"],
            "modality": subset["modality"],
            "image": subset["image_path"].map(lambda p: repo_path(p, prefix)),
            "anat_mask": subset["anat_mask_path"].map(lambda p: repo_path(p, prefix)),
            # The deface mask is optional: "" when an image has none.
            "anon_mask": [
                repo_path(p, prefix) if ok else ""
                for p, ok in zip(anon, _has_path(anon), strict=True)
            ],
            "image_quality_score": subset.get("image_quality_score", float("nan")),
        }
    )
    if manifest["id"].duplicated().any():
        raise ValueError("duplicate volume ids in manifest")
    return manifest[MANIFEST_COLUMNS]


def manifest_files(manifest: pd.DataFrame) -> list[str]:
    """Every repo file the manifest refers to (images and masks)."""
    files: list[str] = []
    for col in ("image", "anat_mask", "anon_mask"):
        files.extend(p for p in manifest[col] if isinstance(p, str) and p)
    return files


def files_to_download(remote_sizes: dict[str, int], local_dir: Path) -> list[tuple[str, int]]:
    """(path, size) pairs not yet on disk. A local file of the right size counts as done."""
    todo = []
    for path, size in remote_sizes.items():
        local = local_dir / path
        if not (local.is_file() and local.stat().st_size == size):
            todo.append((path, size))
    return todo


# The list of files (and sizes) a download run is fetching, so progress can be
# checked from another terminal without touching the network.
PLAN_NAME = "download_plan.csv"


def write_plan(remote_sizes: dict[str, int], plan_path: Path) -> None:
    plan = pd.DataFrame(sorted(remote_sizes.items()), columns=["path", "size"])
    tmp = plan_path.with_name(plan_path.name + ".tmp")
    plan.to_csv(tmp, index=False)
    os.replace(tmp, plan_path)


def download_progress(plan_path: Path, local_dir: Path) -> dict[str, int]:
    """How much of the planned download is on disk (files and bytes).

    Hugging Face writes each file under a temporary name and renames it when
    complete, so a file only counts once it has fully arrived.
    """
    plan = pd.read_csv(plan_path)
    remote = dict(zip(plan["path"], plan["size"], strict=True))
    todo = dict(files_to_download(remote, local_dir))
    return {
        "files_total": len(remote),
        "files_done": len(remote) - len(todo),
        "bytes_total": int(plan["size"].sum()),
        "bytes_done": int(plan["size"].sum()) - sum(todo.values()),
    }


# ---------------------------------------------------------------------------
# Talking to Hugging Face
# ---------------------------------------------------------------------------


def configure_hf_cache(root: Path) -> None:
    """Point Hugging Face's download caches at the SSD. Call before importing huggingface_hub."""
    cache = paths.hf_cache(root)
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(cache / "xet"))


AUTH_HINT = (
    "Could not access the dataset. Check that:\n"
    "  1. you are logged in on this computer:   uv run hf auth login\n"
    "  2. you accepted the dataset's terms (if it asks) at\n"
    "     https://huggingface.co/datasets/{repo_id}"
)


def fetch_metadata(local_dir: Path, repo_id: str = REPO_ID) -> Path:
    """Download openneuro_metadata.csv (small) into `local_dir`."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return Path(
            hf_hub_download(repo_id, METADATA_FILENAME, repo_type="dataset", local_dir=local_dir)
        )
    except RepositoryNotFoundError as e:  # also covers gated / not-logged-in
        raise SystemExit(AUTH_HINT.format(repo_id=repo_id)) from e


def remote_sizes(repo_files: list[str], repo_id: str = REPO_ID, chunk: int = 100) -> dict[str, int]:
    """Size in bytes of each file that exists in the repo (missing files are left out)."""
    from huggingface_hub import HfApi

    api = HfApi()
    sizes: dict[str, int] = {}
    for start in range(0, len(repo_files), chunk):
        batch = repo_files[start : start + chunk]
        for info in api.get_paths_info(repo_id, batch, repo_type="dataset"):
            size = getattr(info, "size", None)
            if size is not None:
                sizes[info.path] = size
    return sizes


def detect_prefix(sample_relative_path: str, repo_id: str = REPO_ID) -> str:
    """Find where the CSV's relative paths live in the repo ("OpenMind/" or the root)."""
    for prefix in (DEFAULT_PREFIX, ""):
        if remote_sizes([repo_path(sample_relative_path, prefix)], repo_id):
            return prefix
    raise FileNotFoundError(
        f"Could not find {sample_relative_path!r} in {repo_id}, "
        f"neither under {DEFAULT_PREFIX}/ nor at the top level."
    )


def download_files(
    todo: list[tuple[str, int]],
    local_dir: Path,
    repo_id: str = REPO_ID,
    workers: int = 8,
    download_fn: Callable[[str], object] | None = None,
) -> list[str]:
    """Download files in parallel. Returns the paths that failed.

    `download_fn(path)` can be swapped out in tests.
    """
    from tqdm import tqdm

    if download_fn is None:
        from huggingface_hub import hf_hub_download

        def download_fn(path: str) -> object:
            return hf_hub_download(repo_id, path, repo_type="dataset", local_dir=local_dir)

    failed: list[str] = []
    total = sum(size for _, size in todo)
    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar,
    ):
        futures = {pool.submit(download_fn, path): (path, size) for path, size in todo}
        for fut in as_completed(futures):
            path, size = futures[fut]
            try:
                fut.result()
            except Exception as e:  # keep going; report at the end
                failed.append(path)
                tqdm.write(f"failed: {path}: {e}")
            bar.update(size)
    return failed
