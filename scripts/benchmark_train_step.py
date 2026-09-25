"""How fast is training (and evaluation) on this machine? Uses made-up clips, no data needed.

    uv run python scripts/benchmark_train_step.py            # batch sizes 4, 8, 16
    uv run python scripts/benchmark_train_step.py --bf16     # also try mixed precision

Loads the pretrained V-JEPA 2.1 ViT-B (downloading it once, like the smoke
test), then times real training steps: teacher + student + predictor,
backward pass, optimiser step. It prints seconds per step, the estimated time
for a 1000-step run, and how long evaluating 200 volumes x 8 clips would take.
It takes a few minutes. If a batch size runs out of memory, it says so and
moves on.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", message=".*sdp_kernel.*", category=FutureWarning)

import torch  # noqa: E402

from please_hold_still import paths  # noqa: E402
from please_hold_still.jepa import JEPA  # noqa: E402
from please_hold_still.masking import ForegroundBlockMasker, token_foreground  # noqa: E402
from please_hold_still.models.vjepa import load_vjepa2_1, pick_device  # noqa: E402
from please_hold_still.training import param_groups  # noqa: E402


def fake_batch(batch_size: int, frames: int, size: int):
    """Random clips with a disk-shaped 'head' taking about 40% of each frame."""
    video = torch.rand(batch_size, 3, frames, size, size) * 2 - 1
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (0.36 * size) ** 2
    return video, disk.expand(batch_size, frames, size, size)


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def gpu_memory_gb(device: torch.device) -> float | None:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return None


def time_train_steps(model, device, batch_size, frames, size, steps, bf16) -> float:
    """Average seconds per training step (after 2 warm-up steps)."""
    optimizer = torch.optim.AdamW(param_groups(model, 0.04), lr=1e-5)
    masker = ForegroundBlockMasker()
    gen = torch.Generator().manual_seed(0)
    video, mask = fake_batch(batch_size, frames, size)
    video = video.to(device)
    fg = token_foreground(mask, model.student.tubelet_size, model.student.patch_size)
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16)
    model.train()
    for i in range(steps + 2):
        if i == 2:
            sync(device)
            t0 = time.time()
        enc, pred = masker(fg, gen)
        enc, pred = [m.to(device) for m in enc], [m.to(device) for m in pred]
        with autocast:
            loss = model(video, enc, pred)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_teacher()
    sync(device)
    return (time.time() - t0) / steps


@torch.inference_mode()
def time_eval_clips(encoder, device, batch_size, frames, size, steps) -> float:
    """Seconds per clip for the frozen encoder (what evaluate_encoder.py does)."""
    video, _ = fake_batch(batch_size, frames, size)
    video = video.to(device)
    encoder.eval()
    for i in range(steps + 1):
        if i == 1:
            sync(device)
            t0 = time.time()
        encoder(video)
    sync(device)
    return (time.time() - t0) / (steps * batch_size)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--steps", type=int, default=5, help="timed steps per setting")
    p.add_argument("--bf16", action="store_true", help="also time bf16 mixed precision")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--data-root", type=Path, default=None, help="default: $PLEASE_HOLD_STILL_DATA")
    p.add_argument("--hub-repo", default=None, help="local clone of facebookresearch/vjepa2")
    p.add_argument("--random-weights", action="store_true", help="skip the checkpoint download")
    p.add_argument("--device", default=None, help="mps / cuda / cpu (default: best available)")
    args = p.parse_args()

    if args.data_root is not None:
        os.environ[paths.ENV_VAR] = str(args.data_root)
    paths.ensure_data_root()
    device = pick_device(args.device)
    encoder, predictor = load_vjepa2_1(
        pretrained=not args.random_weights,
        device="cpu",
        hub_repo=args.hub_repo,
        with_predictor=True,
    )
    model = JEPA(encoder, predictor).to(device)
    print(f"device: {device}\n")

    rows = []
    for bf16 in [False, True] if args.bf16 else [False]:
        for bs in args.batch_sizes:
            label = f"batch {bs:>2}, {'bf16' if bf16 else 'fp32'}"
            try:
                sec = time_train_steps(
                    model, device, bs, args.num_frames, args.size, args.steps, bf16
                )
            except RuntimeError as e:  # out of memory, or bf16 unsupported here
                print(f"{label}: failed ({str(e).splitlines()[0][:90]})")
                continue
            mem = gpu_memory_gb(device)
            rows.append((label, bs, sec))
            print(
                f"{label}: {sec:5.2f} s/step, {bs / sec:5.1f} clips/s, "
                f"1000 steps ~ {1000 * sec / 3600:.1f} h"
                + (f", GPU memory {mem:.0f} GB" if mem is not None else "")
            )
            if device.type == "mps":
                torch.mps.empty_cache()

    per_clip = time_eval_clips(model.teacher, device, 8, args.num_frames, args.size, 3)
    print(
        f"\nevaluation: {per_clip:.2f} s/clip -> 200 volumes x 8 clips ~ "
        f"{1600 * per_clip / 60:.0f} min"
    )
    if rows:
        best = max(rows, key=lambda r: r[1] / r[2])
        print(
            f"fastest per clip: {best[0]} (use --batch-size {best[1]} with train_jepa.py"
            + (", plus --bf16" if "bf16" in best[0] else "")
            + ")"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
