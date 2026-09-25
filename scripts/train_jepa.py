"""Continue V-JEPA 2.1 pretraining on the preprocessed MRI (train studies only).

A first run (prints a time estimate after 5 steps; Ctrl-C saves, re-run resumes):
    caffeinate -i uv run python scripts/train_jepa.py --run-name run1 --steps 1000

Sanity check first: can it memorise a handful of volumes? The loss should fall a lot:
    uv run python scripts/train_jepa.py --run-name overfit --steps 200 \
        --max-volumes 8 --predictor-only-steps 50

Afterwards, evaluate the trained encoder and compare with the original:
    uv run python scripts/evaluate_encoder.py --tag run1 \
        --checkpoint "$PLEASE_HOLD_STILL_DATA/runs/run1/encoder_last.pt"
    uv run python scripts/compare_models.py baseline run1
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", message=".*sdp_kernel.*", category=FutureWarning)

from please_hold_still import paths  # noqa: E402
from please_hold_still.notify import notify  # noqa: E402
from please_hold_still.training import TrainConfig, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        value = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(value, bool):
            p.add_argument(flag, action="store_true", help=f"(default: {value})")
        else:
            kind = float if f.name == "grad_clip" else type(value)
            p.add_argument(flag, type=kind, default=value, help=f"(default: {value})")
    p.add_argument("--data-root", type=Path, default=None, help="default: $PLEASE_HOLD_STILL_DATA")
    p.add_argument("--hub-repo", default=None, help="local clone of facebookresearch/vjepa2")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="start from this Meta-format .pt instead of downloading V-JEPA 2.1",
    )
    p.add_argument("--random-weights", action="store_true", help="untrained network (testing)")
    p.add_argument("--device", default=None, help="mps / cuda / cpu (default: best available)")
    p.add_argument("--max-minutes", type=float, default=None, help="stop (and save) after this")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    root = paths.ensure_data_root(args.data_root)

    result = train(
        cfg,
        root,
        hub_repo=args.hub_repo,
        pretrained_checkpoint=args.checkpoint,
        random_weights=args.random_weights,
        device=args.device,
        max_minutes=args.max_minutes,
    )
    print(f"\n{result['status']} at step {result['step']}. Files in {result['run_dir']}")
    notify("Training " + result["status"].split(":")[0], f"{cfg.run_name}: step {result['step']}")
    return 0 if result["status"] == "finished" else 1


if __name__ == "__main__":
    sys.exit(main())
