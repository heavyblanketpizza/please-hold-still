"""Save a 4x4 grid of one clip's axial slices, anatomy mask overlaid, as a PNG.

    uv run python scripts/visualize_clip.py                 # first volume, central clip
    uv run python scripts/visualize_clip.py --index 5       # sixth volume
    uv run python scripts/visualize_clip.py --random 4      # 4 random volumes/clips

Pictures go to outputs/ (ignored by git) unless you pass --out-dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from mri_jepa import paths
from mri_jepa.data.dataset import MRISliceClipDataset
from mri_jepa.viz import plot_clip_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", type=Path, default=None, help="default: $MRI_JEPA_DATA")
    p.add_argument("--index", type=int, default=0, help="which volume (sorted by id)")
    p.add_argument("--start", type=int, default=None, help="first slice (default: central clip)")
    p.add_argument("--random", type=int, default=0, help="plot N random volumes + clip starts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = paths.ensure_data_root(args.data_root)
    ds = MRISliceClipDataset(
        paths.processed_dir(root), num_frames=16, size=args.size, train=args.random > 0
    )
    if args.random:
        torch.manual_seed(args.seed)
        picks = torch.randperm(len(ds))[: args.random].tolist()
    else:
        picks = [args.index]

    for i in picks:
        item = ds.clip(i, start=args.start)
        title = f"{item['id']}  ({item['modality']})"
        out = args.out_dir / f"{item['id']}_z{item['start']:03d}.png"
        plot_clip_grid(item["video"], item["mask"], out, title=title, first_slice=item["start"])
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
