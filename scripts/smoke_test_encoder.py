"""Smoke test: run one batch of MRI clips through the V-JEPA 2.1 encoder.

    uv run python scripts/smoke_test_encoder.py

Loads the pretrained ViT-B (downloading the checkpoint once, into
<data root>/torch_home), takes one batch from the preprocessed data, and
prints the output token shape. For 16 slices of 256x256 expect
(batch, 2048, 768).

Offline or before any data exists:
    uv run python scripts/smoke_test_encoder.py --random-weights --synthetic
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# Must be set before torch starts: ops the Apple GPU lacks run on the CPU instead.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Deprecation noise from inside Meta's vjepa2 attention code, not ours.
warnings.filterwarnings("ignore", message=".*sdp_kernel.*", category=FutureWarning)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mri_jepa import paths  # noqa: E402
from mri_jepa.data.dataset import MRISliceClipDataset  # noqa: E402
from mri_jepa.models.vjepa import MODELS, load_vjepa2_1_encoder, num_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", type=Path, default=None, help="default: $MRI_JEPA_DATA or SSD")
    p.add_argument("--model", default="vit_base", choices=sorted(MODELS))
    p.add_argument("--hub-repo", default=None, help="local clone of facebookresearch/vjepa2")
    p.add_argument("--checkpoint", type=Path, default=None, help="local .pt instead of download")
    p.add_argument("--random-weights", action="store_true", help="skip the checkpoint")
    p.add_argument("--synthetic", action="store_true", help="random input instead of MRI data")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--device", default=None, help="mps / cuda / cpu (default: best available)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_root is not None:
        os.environ[paths.ENV_VAR] = str(args.data_root)
    root = paths.ensure_data_root()

    # 1. One batch of clips: (B, T, 3, H, W) from the Dataset.
    if args.synthetic:
        video = torch.rand(args.batch_size, args.num_frames, 3, args.size, args.size) * 2 - 1
        ids = ["synthetic"] * args.batch_size
    else:
        ds = MRISliceClipDataset(
            paths.processed_dir(root), num_frames=args.num_frames, size=args.size, train=False
        )
        batch = next(iter(DataLoader(ds, batch_size=args.batch_size)))
        video, ids = batch["video"], batch["id"]
    print(f"clips: {list(ids)}")
    print(f"dataset batch  (B, T, C, H, W): {tuple(video.shape)}")

    # 2. The encoder wants channels before time: (B, C, T, H, W).
    x = video.permute(0, 2, 1, 3, 4).contiguous()
    print(f"encoder input  (B, C, T, H, W): {tuple(x.shape)}")

    # 3. Load the model and run it.
    t0 = time.time()
    encoder = load_vjepa2_1_encoder(
        args.model,
        pretrained=not args.random_weights,
        device=args.device,
        hub_repo=args.hub_repo,
        checkpoint=args.checkpoint,
    )
    device = next(encoder.parameters()).device
    print(
        f"loaded {args.model} ({'random' if args.random_weights else 'pretrained'} weights) "
        f"on {device} in {time.time() - t0:.1f}s"
    )

    t0 = time.time()
    with torch.inference_mode():
        tokens = encoder(x.to(device))
    if device.type == "mps":
        torch.mps.synchronize()
    print(f"forward pass: {time.time() - t0:.2f}s")

    expected = num_tokens(args.num_frames, args.size, args.size, args.model)
    print(f"\noutput tokens  (B, N, D): {tuple(tokens.shape)}   (expected N = {expected})")
    print(
        f"token stats: mean {tokens.mean().item():+.3f}, std {tokens.std().item():.3f}, "
        f"finite: {bool(torch.isfinite(tokens).all())}"
    )
    return 0 if tokens.shape[1] == expected and torch.isfinite(tokens).all() else 1


if __name__ == "__main__":
    sys.exit(main())
