"""Probe a frozen encoder: how much does it know about the scans?

"Before" (the original V-JEPA 2.1 ViT-B), run once before any training:
    uv run python scripts/evaluate_encoder.py --tag baseline

"After" (an encoder snapshot saved by train_jepa.py):
    uv run python scripts/evaluate_encoder.py --tag run1 \
        --checkpoint "$MRI_JEPA_DATA/runs/run1/encoder_last.pt"

Then compare:
    uv run python scripts/compare_models.py baseline run1

Results go to <data root>/eval/<tag>/ (features, clip table, results.json).
It prints a time estimate after the first batch. With ~200 volumes x 8 clips
it should take a few minutes on the Mac's GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", message=".*sdp_kernel.*", category=FutureWarning)

import numpy as np  # noqa: E402

from mri_jepa import paths  # noqa: E402
from mri_jepa.data.openmind import METADATA_FILENAME  # noqa: E402
from mri_jepa.models.vjepa import MODELS, load_vjepa2_1_encoder  # noqa: E402
from mri_jepa.notify import notify  # noqa: E402
from mri_jepa.probe import (  # noqa: E402
    attach_metadata_labels,
    comparison_table,
    extract_features,
    run_probes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tag", required=True, help="name for this model, e.g. baseline or run1")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="encoder .pt from training (default: original pretrained weights)",
    )
    p.add_argument("--data-root", type=Path, default=None, help="default: $MRI_JEPA_DATA")
    p.add_argument("--model", default="vit_base", choices=sorted(MODELS))
    p.add_argument("--hub-repo", default=None, help="local clone of facebookresearch/vjepa2")
    p.add_argument("--random-weights", action="store_true", help="untrained network (testing)")
    p.add_argument("--clips-per-volume", type=int, default=8)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None, help="mps / cuda / cpu (default: best available)")
    p.add_argument("--overwrite", action="store_true", help="redo features for an existing tag")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = paths.ensure_data_root(args.data_root)
    out = paths.eval_dir(root) / args.tag
    features_path, clips_path = out / "features.npy", out / "clips.csv"

    if features_path.is_file() and clips_path.is_file() and not args.overwrite:
        import pandas as pd

        print(f"reusing features in {out} (pass --overwrite to recompute)")
        features, clips = np.load(features_path), pd.read_csv(clips_path)
    else:
        encoder = load_vjepa2_1_encoder(
            args.model,
            pretrained=not args.random_weights,
            device=args.device,
            hub_repo=args.hub_repo,
            checkpoint=args.checkpoint,
        )
        features, clips = extract_features(
            encoder,
            paths.processed_dir(root),
            clips_per_volume=args.clips_per_volume,
            num_frames=args.num_frames,
            size=args.size,
            batch_size=args.batch_size,
        )
        out.mkdir(parents=True, exist_ok=True)
        np.save(features_path, features)
        clips.to_csv(clips_path, index=False)

    metadata_csv = paths.raw_dir(root) / METADATA_FILENAME
    if metadata_csv.is_file():
        clips = attach_metadata_labels(clips, metadata_csv)
    probes = run_probes(features, clips)

    summary = {
        "tag": args.tag,
        "checkpoint": str(args.checkpoint) if args.checkpoint else "original pretrained",
        "n_volumes": int(clips["id"].nunique()),
        "n_clips": int(len(clips)),
        "n_test_volumes": int(clips.loc[clips["split"] == "test", "id"].nunique()),
        "probes": probes,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{summary['n_volumes']} volumes ({summary['n_test_volumes']} held out for testing)")
    table = comparison_table({args.tag: probes})
    print(table.to_string(index=False) if len(table) else "no probe could run")
    for task, r in probes.items():
        if "skipped" in r:
            print(f"  {task}: skipped ({r['skipped']})")
    print(f"\nsaved {out / 'results.json'}")
    notify("Evaluation finished", f"{args.tag}: {len(table)} probes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
