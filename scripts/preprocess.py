"""Preprocess the downloaded OpenMind subset into normalised .npy arrays.

Reads <data root>/raw/subset_manifest.csv (written by download_openmind.py)
and writes to <data root>/processed/. Safe to stop and re-run: finished
volumes are skipped.

Try a few volumes first to see how long one takes:
    uv run python scripts/preprocess.py --limit 5
Then everything:
    uv run python scripts/preprocess.py --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

from mri_jepa import paths
from mri_jepa.data.openmind import MANIFEST_NAME
from mri_jepa.data.preprocess import PreprocessConfig, process_many
from mri_jepa.notify import notify


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", type=Path, default=None, help="default: $MRI_JEPA_DATA")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--limit", type=int, default=None, help="only the first N volumes")
    p.add_argument("--overwrite", action="store_true", help="redo finished volumes")
    p.add_argument("--spacing", type=float, default=1.0, help="target voxel size in mm")
    p.add_argument("--margin", type=int, default=0, help="voxels kept around the anatomy box")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = paths.ensure_data_root(args.data_root)
    raw, out = paths.raw_dir(root), paths.processed_dir(root)
    manifest_path = raw / MANIFEST_NAME
    if not manifest_path.is_file():
        print(f"{manifest_path} not found. Run scripts/download_openmind.py first.")
        return 1
    out.mkdir(exist_ok=True)

    rows = pd.read_csv(manifest_path).to_dict("records")
    if args.limit:
        rows = rows[: args.limit]
    cfg = PreprocessConfig(spacing_mm=args.spacing, margin=args.margin)
    print(f"preprocessing {len(rows)} volumes with {args.workers} workers -> {out}")

    start = time.time()
    results = process_many(rows, raw, out, cfg, args.workers, args.overwrite)
    minutes = (time.time() - start) / 60

    status = pd.DataFrame(results, columns=["id", "status", "message"])
    print(f"\ndone in {minutes:.1f} min: {status['status'].value_counts().to_dict()}")
    failed = status[status["status"] == "failed"]
    failures_csv = out / "_failures.csv"
    if len(failed):
        failed.to_csv(failures_csv, index=False)
        print(f"{len(failed)} failures listed in {failures_csv}, e.g.:")
        for _, r in failed.head(5).iterrows():
            print(f"  {r['id']}: {r['message']}")
    elif failures_csv.exists():
        failures_csv.unlink()  # stale list from an earlier run
    counts = status["status"].value_counts().to_dict()
    notify("Preprocessing finished", ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
