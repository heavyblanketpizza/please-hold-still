"""One step of continued V-JEPA 2.1 pretraining: student, EMA teacher, predictor, loss.

How a step works (follows Meta's app/vjepa_2_1/train.py at VJEPA_COMMIT):

1. The *teacher* (a frozen copy of the encoder) sees the whole clip. Its
   output tokens, layer-normalised, are the targets.
2. The *student* encoder sees only the context tokens (the rest are hidden by
   the masks from `please_hold_still.masking`).
3. The *predictor* takes the student's tokens and guesses the teacher's
   tokens at the hidden positions ("target predictions"). It also re-predicts
   the visible ones ("context predictions", V-JEPA 2.1's dense loss).
4. loss = L1(target predictions) + lambda * L1(context predictions). Each
   context token is weighted by 1/sqrt(distance to the nearest hidden token),
   so tokens right next to a hole count most.
5. After the optimiser step, `update_teacher()` moves the teacher a tiny bit
   toward the student (exponential moving average, EMA).

Differences from Meta's from-scratch recipe:
- The released ViT-B was distilled from ViT-G, so its predictor outputs
  1664-dim ViT-G features. Our teacher is the ViT-B itself (768-dim), so the
  predictor's two output layers are replaced with fresh ones. Everything else
  in the predictor keeps its pretrained weights.
- Targets are the teacher's last layer only (like the released checkpoint),
  not the 4-layer "deep supervision" used when training from scratch.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from please_hold_still.models.vjepa import set_predictor_grid

EMA_MOMENTUM = 0.99925  # V-JEPA 2.1 configs
CONTEXT_LOSS_WEIGHT = 0.5  # lambda_value_vid in V-JEPA 2.1 configs


def gather_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """tokens (B, N, D), idx (B, K) -> (B, K, D)."""
    return torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))


def token_coords(idx: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    """Flat token indices (B, K) -> (t, row, col) coordinates (B, K, 3) as floats."""
    _, height, width = grid
    t = idx // (height * width)
    rest = idx % (height * width)
    return torch.stack([t, rest // width, rest % width], dim=-1).float()


def context_distance_weights(
    masks_enc: torch.Tensor, masks_pred: torch.Tensor, grid: tuple[int, int, int]
) -> torch.Tensor:
    """1/sqrt(distance from each context token to its nearest target token), (B, K).

    Context and targets never overlap, so every distance is at least 1.
    """
    dist = torch.cdist(token_coords(masks_enc, grid), token_coords(masks_pred, grid))
    return dist.amin(dim=-1).clamp(min=1.0).rsqrt()


def replace_prediction_heads(predictor: nn.Module, out_dim: int) -> None:
    """Give the predictor fresh output layers that produce `out_dim` features."""
    in_dim = predictor.predictor_proj.in_features
    for name in ("predictor_proj", "predictor_proj_context"):
        layer = nn.Linear(in_dim, out_dim)
        nn.init.trunc_normal_(layer.weight, std=0.02)
        nn.init.zeros_(layer.bias)
        setattr(predictor, name, layer)


class JEPA(nn.Module):
    """Student encoder + EMA teacher + predictor for continued pretraining."""

    def __init__(self, encoder: nn.Module, predictor: nn.Module, ema_momentum=EMA_MOMENTUM):
        super().__init__()
        self.student = encoder
        self.teacher = copy.deepcopy(encoder).requires_grad_(False)
        self.predictor = predictor
        self.ema_momentum = ema_momentum
        replace_prediction_heads(predictor, encoder.embed_dim)

    def train(self, mode: bool = True) -> JEPA:
        super().train(mode)
        self.teacher.eval()  # the teacher is never in training mode
        return self

    def forward(
        self,
        video: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
        context_weight: float = CONTEXT_LOSS_WEIGHT,
    ) -> dict[str, torch.Tensor]:
        """video (B, 3, T, H, W); one (B, K) context and target index tensor per mask type."""
        _, _, frames, height, width = video.shape
        if height != width:
            raise ValueError(f"frames must be square, got {height}x{width}")
        patch, tubelet = self.student.patch_size, self.student.tubelet_size
        grid = (frames // tubelet, height // patch, width // patch)
        set_predictor_grid(self.predictor, height // patch, frames, tubelet)

        with torch.no_grad():
            targets = self.teacher(video)
            targets = F.layer_norm(targets, (targets.size(-1),))

        loss_pred = loss_ctx = video.new_zeros(())
        for m_enc, m_pred in zip(masks_enc, masks_pred, strict=True):
            z = self.student(video, masks=m_enc)
            # mask_index=0: Meta picks the mask token by clip-length group; we have one.
            z_pred, z_ctx = self.predictor(z, m_enc, m_pred, mask_index=0)
            loss_pred = loss_pred + (z_pred - gather_tokens(targets, m_pred)).abs().mean()
            weights = context_distance_weights(m_enc, m_pred, grid).unsqueeze(-1)
            loss_ctx = loss_ctx + ((z_ctx - gather_tokens(targets, m_enc)).abs() * weights).mean()
        loss_pred = loss_pred / len(masks_enc)
        loss_ctx = loss_ctx / len(masks_enc)
        return {
            "loss": loss_pred + context_weight * loss_ctx,
            "loss_pred": loss_pred.detach(),
            "loss_context": loss_ctx.detach(),
        }

    @torch.no_grad()
    def update_teacher(self, momentum: float | None = None) -> None:
        """teacher <- m * teacher + (1 - m) * student."""
        m = self.ema_momentum if momentum is None else momentum
        t_params = list(self.teacher.parameters())
        s_params = list(self.student.parameters())
        torch._foreach_mul_(t_params, m)
        torch._foreach_add_(t_params, s_params, alpha=1.0 - m)
