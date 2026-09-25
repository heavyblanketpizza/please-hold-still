"""Download a subset of the OpenMind dataset from Hugging Face.

Step 1 (a few MB) - look at the metadata table only:
    uv run python scripts/download_openmind.py --inspect

Step 2 - pick and download ~200 T1w/T2w/FLAIR volumes plus their masks:
    uv run python scripts/download_openmind.py --n 200

Scale up later with a bigger --n. The pick for a bigger n includes the smaller
one, so files already downloaded are reused. The script prints the total size
first and stops if it is above --max-gb (default 5).

Files land in <data root>/raw/, which defaults to
"/Volumes/Just for Fun/mri-jepa-data/raw". Re-running skips finished files.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from mri_jepa import paths
from mri_jepa.data import openmind as om

GB = 1024**3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", type=Path, default=None, help="default: $MRI_JEPA_DATA or SSD")
    p.add_argument("--repo-id", default=om.REPO_ID)
    p.add_argument("--inspect", action="store_true", help="only fetch + summarise the CSV")
    p.add_argument("--n", type=int, default=200, help="total number of volumes (default 200)")
    p.add_argument("--modalities", nargs="+", default=list(om.DEFAULT_MODALITIES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iqs", type=float, default=None, help="skip image_quality_score above")
    p.add_argument("--max-gb", type=float, default=5.0, help="refuse bigger downloads (GB)")
    p.add_argument("--workers", type=int, default=8, help="parallel downloads")
    p.add_argument("--dry-run", action="store_true", help="plan and print sizes, download nothing")
    return p.parse_args()


def summarise(meta) -> None:
    print(f"\n{len(meta):,} images from {meta['dataset_id'].nunique()} studies")
    print(f"columns: {list(meta.columns)}\n")
    print("images per modality (top 25):")
    print(meta["modality"].value_counts().head(25).to_string())
    for col in ("anat_mask_path", "anon_mask_path"):
        if col in meta:
            print(f"\nrows with {col}: {om._has_path(meta[col]).sum():,}")
    if "image_quality_score" in meta:
        print("\nimage_quality_score (lower is better):")
        print(meta["image_quality_score"].describe().to_string())
    print("\nfirst 3 rows:")
    print(meta.head(3).T.to_string())


def main() -> int:
    args = parse_args()
    root = paths.ensure_data_root(args.data_root)
    om.configure_hf_cache(root)
    raw = paths.raw_dir(root)
    raw.mkdir(exist_ok=True)

    csv_path = raw / om.METADATA_FILENAME
    if not csv_path.is_file():
        print(f"fetching {om.METADATA_FILENAME} ...")
        om.fetch_metadata(raw, args.repo_id)
    meta = om.load_metadata(csv_path)

    if args.inspect:
        summarise(meta)
        prefix = om.detect_prefix(meta["image_path"].iloc[0], args.repo_id)
        print(f"\nimage files live under: {prefix + '/' if prefix else '(repo root)'}")
        return 0

    subset = om.select_subset(meta, args.n, args.modalities, args.seed, args.max_iqs)
    print(f"selected {len(subset)} images from {subset['dataset_id'].nunique()} studies:")
    print(subset["modality"].value_counts().to_string())

    prefix = om.detect_prefix(subset["image_path"].iloc[0], args.repo_id)
    manifest = om.build_manifest(subset, prefix)
    wanted = om.manifest_files(manifest)
    print(f"\nlooking up sizes of {len(wanted)} files ...")
    sizes = om.remote_sizes(wanted, args.repo_id)
    missing_remote = sorted(set(wanted) - set(sizes))
    if missing_remote:
        print(f"warning: {len(missing_remote)} files listed in the CSV are not in the repo, e.g.")
        print("  " + "\n  ".join(missing_remote[:5]))

    todo = om.files_to_download(sizes, raw)
    todo_bytes = sum(s for _, s in todo)
    free = shutil.disk_usage(raw).free
    print(
        f"\ntotal {sum(sizes.values()) / GB:.2f} GB in {len(sizes)} files; "
        f"still to download: {todo_bytes / GB:.2f} GB in {len(todo)} files; "
        f"free on drive: {free / GB:.1f} GB"
    )
    if todo_bytes > args.max_gb * GB:
        print(f"stopping: more than --max-gb {args.max_gb}. Re-run with a larger --max-gb.")
        return 2
    if todo_bytes > free * 0.9:
        print("stopping: not enough free space on the drive.")
        return 2
    if args.dry_run:
        return 0

    failed = om.download_files(todo, raw, args.repo_id, args.workers) if todo else []

    # Keep only volumes whose image and masks are all on disk.
    def complete(row) -> bool:
        return all(
            (raw / p).is_file()
            for p in (row["image"], row["anat_mask"], row["anon_mask"])
            if isinstance(p, str) and p
        )

    ok = manifest[manifest.apply(complete, axis=1)]
    ok.to_csv(raw / om.MANIFEST_NAME, index=False)
    print(f"\nwrote {raw / om.MANIFEST_NAME} with {len(ok)} of {len(manifest)} volumes")
    if failed:
        print(f"{len(failed)} downloads failed; re-run the same command to retry them.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
