"""The JEPA training step.

Pure helpers are tested always. Tests with the real V-JEPA architecture need
a local clone of facebookresearch/vjepa2 (see tests/test_vjepa.py) and use
tiny clips (4 frames x 64 x 64 = 2 x 4 x 4 tokens) so they run in seconds.
"""

import os

import pytest
import torch

from mri_jepa import jepa
from mri_jepa.masking import ForegroundBlockMasker, token_foreground

VJEPA2_REPO = os.environ.get("VJEPA2_REPO")
needs_repo = pytest.mark.skipif(not VJEPA2_REPO, reason="set VJEPA2_REPO to a vjepa2 clone")


def test_token_coords():
    idx = torch.tensor([[0, 17, 300]])
    coords = jepa.token_coords(idx, (8, 16, 16)).tolist()[0]
    assert coords == [[0, 0, 0], [0, 1, 1], [1, 2, 12]]


def test_context_distance_weights():
    grid = (1, 1, 10)  # one row of 10 tokens
    ctx = torch.tensor([[0, 3, 4]])
    tgt = torch.tensor([[5, 6]])
    w = jepa.context_distance_weights(ctx, tgt, grid)[0]
    # distances to the nearest target: 5, 2, 1 -> weights 1/sqrt(d)
    assert torch.allclose(w, torch.tensor([5.0, 2.0, 1.0]).rsqrt())


def test_gather_tokens():
    tokens = torch.arange(2 * 5 * 3).float().view(2, 5, 3)
    out = jepa.gather_tokens(tokens, torch.tensor([[4, 0], [1, 1]]))
    assert torch.equal(out[0, 0], tokens[0, 4]) and torch.equal(out[1, 1], tokens[1, 1])


# --- with the real architecture ---------------------------------------------


@pytest.fixture
def model():
    from mri_jepa.models.vjepa import load_vjepa2_1

    torch.manual_seed(0)
    enc, pred = load_vjepa2_1(
        pretrained=False, device="cpu", hub_repo=VJEPA2_REPO, with_predictor=True
    )
    return jepa.JEPA(enc, pred)


def tiny_batch(b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    video = torch.randn(b, 3, 4, 64, 64, generator=g)
    mask = torch.zeros(b, 4, 64, 64, dtype=torch.bool)
    mask[:, :, 8:56, 8:56] = True
    masks_enc, masks_pred = ForegroundBlockMasker(drop_background=False)(token_foreground(mask), g)
    return video, masks_enc, masks_pred


@needs_repo
def test_prediction_heads_match_teacher_width(model):
    assert model.predictor.predictor_proj.out_features == 768
    assert model.predictor.predictor_proj_context.out_features == 768


@needs_repo
def test_predictor_uses_the_real_token_grid(model):
    from mri_jepa.models.vjepa import set_predictor_grid

    attn = model.predictor.predictor_blocks[0].attn
    set_predictor_grid(model.predictor, 16, 16)
    t, h, w = attn.separate_positions(torch.tensor([[300]]))
    assert (int(t), int(h), int(w)) == (1, 2, 12)
    model(*tiny_batch())  # forward adapts the grid to the clip: 64 px -> 4 x 4
    assert attn.grid_size == 4 and model.predictor.num_patches == 2 * 4 * 4


@needs_repo
def test_loss_and_gradients(model):
    out = model(*tiny_batch())
    assert set(out) == {"loss", "loss_pred", "loss_context"}
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    out["loss"].backward()
    assert any(p.grad is not None for p in model.student.parameters())
    assert model.predictor.predictor_proj.weight.grad is not None
    assert all(p.grad is None for p in model.teacher.parameters())  # teacher never learns


@needs_repo
def test_teacher_stays_in_eval_mode(model):
    model.train()
    assert model.student.training and not model.teacher.training


@needs_repo
def test_ema_update(model):
    with torch.no_grad():
        for p in model.student.parameters():
            p.add_(1.0)  # pretend an optimiser step moved the student
    before = [p.clone() for p in model.teacher.parameters()]
    model.update_teacher(momentum=0.9)
    for t_old, t_new, s in zip(
        before, model.teacher.parameters(), model.student.parameters(), strict=True
    ):
        assert torch.allclose(t_new, 0.9 * t_old + 0.1 * s, atol=1e-6)


@needs_repo
def test_a_few_steps_reduce_the_loss(model):
    """Sanity check: student + predictor learn on one fixed batch.

    The clip is random noise, so hidden tokens are basically unpredictable and
    their loss falls only slowly; the visible (context) tokens can be
    re-predicted, so that loss must fall fast.
    """
    batch = tiny_batch()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4)
    model.train()
    history = []
    for _ in range(30):
        out = model(*batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        model.update_teacher()
        history.append((out["loss_pred"].item(), out["loss_context"].item()))
    (pred_0, ctx_0), (pred_n, ctx_n) = history[0], history[-1]
    assert ctx_n < 0.6 * ctx_0
    assert pred_n < pred_0
