"""Which tokens the student sees and which it must predict (V-JEPA multi-block masks).

A clip of 16 slices x 256 x 256 becomes a grid of 8 x 16 x 16 tokens (time,
row, column). Each token covers 2 slices x 16 x 16 pixels. Token i sits at
flat index t*16*16 + h*16 + w, the order in which the encoder flattens them.

V-JEPA 2.1 recipe (configs/train_2_1/*/pretrain-256px-16f.yaml), kept here:
- two mask types per batch: 8 small blocks (short-range) and 2 large blocks
  (long-range);
- every block spans all time steps (here, all 16 slices), so the model cannot
  just copy the same spot from a neighbouring slice;
- hidden tokens = prediction targets; the rest = visible context.

Changes for MRI:
- blocks are centred on tokens that contain anatomy, so we never ask the
  model to predict empty air;
- block area is a fraction of the head's cross-section, not the whole frame
  (a 70% block hides ~70% of the head rather than all of it);
- pure-air tokens are dropped from both context and targets (saves compute);
- to give every sample in a batch the same number of indices, longer lists
  are randomly thinned. Meta cuts off the end instead, which would always
  drop the top slices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BlockMaskConfig:
    num_blocks: int
    spatial_scale: float  # block area, as a fraction of the head's cross-section
    temporal_scale: float = 1.0  # fraction of time steps a block spans (1.0 = all)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)


VJEPA21_MASKS = (BlockMaskConfig(8, 0.15), BlockMaskConfig(2, 0.7))


def token_foreground(mask: torch.Tensor, tubelet: int = 2, patch: int = 16) -> torch.Tensor:
    """(B, T, H, W) anatomy mask -> (B, T/tubelet, H/patch, W/patch) anatomy fraction per token."""
    x = mask.float().unsqueeze(1)
    return F.avg_pool3d(x, kernel_size=(tubelet, patch, patch)).squeeze(1)


def _rand(g: torch.Generator | None) -> float:
    return torch.rand(1, generator=g).item()


def _randint(high: int, g: torch.Generator | None) -> int:
    """Uniform integer in [0, high)."""
    return int(torch.randint(high, (1,), generator=g).item())


def sample_one(
    fg: torch.Tensor,
    cfg: BlockMaskConfig,
    g: torch.Generator | None = None,
    drop_background: bool = True,
    max_attempts: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Context and target token indices for one clip.

    fg: (t, h, w) bool, True where the token contains anatomy.
    Returns two sorted 1D LongTensors of flat token indices, never overlapping
    and never empty.
    """
    if not fg.any():  # e.g. a clip of padding only: treat everything as foreground
        fg = torch.ones_like(fg)
    duration, height, width = fg.shape
    centres = fg.any(0).nonzero()  # (row, col) positions with anatomy in some slice
    head_area = len(centres)

    for _ in range(max_attempts):
        hidden = torch.zeros_like(fg)
        for _ in range(cfg.num_blocks):
            area = max(1.0, cfg.spatial_scale * head_area)
            lo, hi = cfg.aspect_ratio
            ar = lo + _rand(g) * (hi - lo)
            bh = min(height, max(1, round(math.sqrt(area * ar))))
            bw = min(width, max(1, round(math.sqrt(area / ar))))
            bt = max(1, int(duration * cfg.temporal_scale))
            cy, cx = centres[_randint(head_area, g)].tolist()
            top = min(max(cy - bh // 2, 0), height - bh)
            left = min(max(cx - bw // 2, 0), width - bw)
            start = _randint(duration - bt + 1, g)
            hidden[start : start + bt, top : top + bh, left : left + bw] = True
        target = hidden & fg
        context = ~hidden & fg if drop_background else ~hidden
        if context.any() and target.any():
            break
    else:
        # Blocks kept covering everything (tiny head section): reveal ~10% of the
        # targets as context so the student always sees something.
        idx = target.flatten().nonzero().squeeze(1)
        reveal = idx[torch.randperm(len(idx), generator=g)[: max(1, len(idx) // 10)]]
        target = target.flatten().clone()
        context = context.flatten().clone()
        target[reveal], context[reveal] = False, True
    return context.flatten().nonzero().squeeze(1), target.flatten().nonzero().squeeze(1)


def equalize(index_lists: list[torch.Tensor], g: torch.Generator | None = None) -> torch.Tensor:
    """Stack index lists into (B, K), K = shortest length, randomly thinning longer ones."""
    k = min(len(ix) for ix in index_lists)
    rows = [ix[torch.randperm(len(ix), generator=g)[:k]].sort().values for ix in index_lists]
    return torch.stack(rows)


class ForegroundBlockMasker:
    """Make V-JEPA context/target masks for a batch, restricted to anatomy.

    Call with the per-token anatomy fraction (see `token_foreground`). Returns
    (masks_enc, masks_pred): one (B, K) LongTensor per mask type, the format
    the V-JEPA 2.1 encoder and predictor take.
    """

    def __init__(
        self,
        configs: tuple[BlockMaskConfig, ...] = VJEPA21_MASKS,
        min_fg_fraction: float = 0.0,
        drop_background: bool = True,
    ):
        self.configs = configs
        self.min_fg_fraction = min_fg_fraction  # a token counts as anatomy above this
        self.drop_background = drop_background

    def __call__(
        self, token_fg: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        fg = token_fg > self.min_fg_fraction
        masks_enc, masks_pred = [], []
        for cfg in self.configs:
            pairs = [sample_one(f, cfg, generator, self.drop_background) for f in fg]
            masks_enc.append(equalize([c for c, _ in pairs], generator))
            masks_pred.append(equalize([t for _, t in pairs], generator))
        return masks_enc, masks_pred
