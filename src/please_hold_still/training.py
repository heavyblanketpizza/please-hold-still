"""Continued V-JEPA 2.1 pretraining on MRI clips: the training loop.

Each step: take a batch of clips from the *train* studies, hide blocks of
anatomy (`ForegroundBlockMasker`), compute the JEPA loss (`JEPA`), update the
student and predictor with AdamW, then move the teacher toward the student
(EMA).

Phases:
- predictor-only warm-up (`predictor_only_steps`): the student encoder is
  frozen while the predictor, whose output layers are brand new, catches
  up. Otherwise random gradients from the new layers would disturb the
  pretrained encoder.
- full training: student and predictor learn together.

Learning rate: linear warm-up, then cosine decay to `final_lr_fraction * lr`.

Everything is saved under <data root>/runs/<run_name>/:
    config.json          the settings used
    log.csv              loss and learning rate every `log_every` steps
    checkpoint_last.pt   full state for resuming (student, teacher, predictor, optimiser)
    encoder_last.pt      teacher weights in Meta's checkpoint format ("ema_encoder"),
    encoder_stepN.pt     loadable by `load_vjepa2_1_encoder(checkpoint=...)`
Re-running the same command resumes from checkpoint_last.pt. Ctrl-C saves first.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from please_hold_still import paths
from please_hold_still.data.dataset import MRISliceClipDataset
from please_hold_still.jepa import CONTEXT_LOSS_WEIGHT, EMA_MOMENTUM, JEPA
from please_hold_still.masking import ForegroundBlockMasker, token_foreground
from please_hold_still.models.vjepa import load_vjepa2_1, pick_device


@dataclass
class TrainConfig:
    run_name: str = "run1"
    steps: int = 1000
    batch_size: int = 8
    lr: float = 1e-4
    final_lr_fraction: float = 0.1
    warmup_steps: int = 100
    predictor_only_steps: int = 200
    weight_decay: float = 0.04
    ema_momentum: float = EMA_MOMENTUM
    context_weight: float = CONTEXT_LOSS_WEIGHT
    grad_clip: float | None = None
    num_frames: int = 16
    size: int = 256
    num_workers: int = 4
    save_every: int = 250
    log_every: int = 10
    seed: int = 0
    max_volumes: int = 0  # 0 = all; e.g. 8 for an "can it memorise?" sanity check
    bf16: bool = False  # mixed precision; faster, try once plain float32 works


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate for a 0-based step: linear warm-up, then cosine decay."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    span = max(cfg.steps - cfg.warmup_steps, 1)
    progress = min((step - cfg.warmup_steps) / span, 1.0)
    final = cfg.lr * cfg.final_lr_fraction
    return final + (cfg.lr - final) * 0.5 * (1 + math.cos(math.pi * progress))


def param_groups(model: JEPA, weight_decay: float) -> list[dict]:
    """Student + predictor parameters; no weight decay on biases and norm layers (as Meta)."""
    decay, no_decay = [], []
    for module in (model.student, model.predictor):
        for p in module.parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def disk_needed(model: JEPA, cfg: TrainConfig) -> int:
    """Bytes a run needs: the resume checkpoint (twice, while it is replaced) + snapshots."""
    trainable = sum(p.numel() for m in (model.student, model.predictor) for p in m.parameters())
    everything = sum(p.numel() for p in model.parameters())
    checkpoint = 4 * (everything + 2 * trainable)  # float32 weights + AdamW's 2 running averages
    snapshot = 4 * sum(p.numel() for p in model.teacher.parameters())
    return 2 * checkpoint + snapshot * (cfg.steps // cfg.save_every + 2)


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _batches(loader: DataLoader):
    """Loop over the loader forever (a new random clip per volume each pass)."""
    while True:
        yield from loader


def train(
    cfg: TrainConfig,
    root: Path,
    hub_repo: str | Path | None = None,
    pretrained_checkpoint: str | Path | None = None,
    random_weights: bool = False,
    device: str | None = None,
    max_minutes: float | None = None,
    progress_print=print,
) -> dict:
    """Run (or resume) training. Returns a small summary dict."""
    torch.manual_seed(cfg.seed)
    run_dir = paths.runs_dir(root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint_last.pt"

    # -- data: train studies only; the test studies are kept for the probes
    ds = MRISliceClipDataset(
        paths.processed_dir(root), cfg.num_frames, cfg.size, train=True, split="train"
    )
    if cfg.max_volumes:
        ds.items = ds.items[: cfg.max_volumes]
    batch_size = min(cfg.batch_size, len(ds))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # -- model
    encoder, predictor = load_vjepa2_1(
        pretrained=not random_weights,
        device="cpu",
        hub_repo=hub_repo,
        checkpoint=pretrained_checkpoint,
        with_predictor=True,
    )
    model = JEPA(encoder, predictor, ema_momentum=cfg.ema_momentum)
    dev = pick_device(device)
    model = model.to(dev)
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)

    need, free = disk_needed(model, cfg), shutil.disk_usage(run_dir).free
    progress_print(f"checkpoints will need about {need / 1e9:.1f} GB ({free / 1e9:.0f} GB free)")
    if need > free:
        raise RuntimeError("not enough free disk space for this run's checkpoints")

    start = 0
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
        progress_print(f"resuming {cfg.run_name} from step {start}")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    def save(step: int, snapshot: bool) -> None:
        _atomic_torch_save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
            ckpt_path,
        )
        # Meta's format: loadable by load_vjepa2_1_encoder(checkpoint=...)
        encoder_state = {"ema_encoder": model.teacher.state_dict(), "step": step}
        _atomic_torch_save(encoder_state, run_dir / "encoder_last.pt")
        if snapshot:
            _atomic_torch_save(encoder_state, run_dir / f"encoder_step{step}.pt")

    log_path = run_dir / "log.csv"
    new_log = not log_path.is_file()
    log_file = open(log_path, "a", newline="")  # noqa: SIM115 (closed in finally)
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "loss", "loss_pred", "loss_context", "lr", "sec_per_step", "phase"])

    masker = ForegroundBlockMasker()
    mask_gen = torch.Generator().manual_seed(cfg.seed + start)
    autocast = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=cfg.bf16)
    model.train()
    batches = _batches(loader)
    step, t_start = start, time.time()
    status = "finished"
    try:
        while step < cfg.steps:
            warmup_phase = step < cfg.predictor_only_steps
            model.student.requires_grad_(not warmup_phase)
            lr = lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            batch = next(batches)
            video = batch["video"].permute(0, 2, 1, 3, 4).to(dev)
            fg = token_foreground(batch["mask"], encoder.tubelet_size, encoder.patch_size)
            masks_enc, masks_pred = masker(fg, mask_gen)
            masks_enc = [m.to(dev) for m in masks_enc]
            masks_pred = [m.to(dev) for m in masks_pred]

            with autocast:
                out = model(video, masks_enc, masks_pred, context_weight=cfg.context_weight)
            loss = out["loss"]
            if not torch.isfinite(loss):
                status = "stopped: loss is not finite (try a lower --lr)"
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], cfg.grad_clip
                )
            optimizer.step()
            if not warmup_phase:
                model.update_teacher()
            step += 1

            if step % cfg.log_every == 0 or step == cfg.steps or step == start + 5:
                sec = (time.time() - t_start) / max(step - start, 1)
                log.writerow(
                    [
                        step,
                        f"{loss.item():.5f}",
                        f"{out['loss_pred'].item():.5f}",
                        f"{out['loss_context'].item():.5f}",
                        f"{lr:.3e}",
                        f"{sec:.3f}",
                        "predictor-only" if warmup_phase else "full",
                    ]
                )
                log_file.flush()
                eta = (cfg.steps - step) * sec / 60
                progress_print(
                    f"step {step}/{cfg.steps}  loss {loss.item():.4f} "
                    f"(pred {out['loss_pred'].item():.4f}, ctx {out['loss_context'].item():.4f})"
                    f"  lr {lr:.2e}  {sec:.2f}s/step  ~{eta:.0f} min left"
                    + ("  [predictor-only warm-up]" if warmup_phase else "")
                )
            if step % cfg.save_every == 0:
                save(step, snapshot=True)
            if max_minutes is not None and (time.time() - t_start) / 60 > max_minutes:
                status = f"stopped after --max-minutes {max_minutes}"
                break
    except KeyboardInterrupt:
        status = "interrupted (Ctrl-C)"
    finally:
        if step > start:
            save(step, snapshot=step == cfg.steps)
        log_file.close()
    return {"status": status, "step": step, "run_dir": str(run_dir)}
