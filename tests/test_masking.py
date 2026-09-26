"""Foreground-aware V-JEPA block masks."""

import torch

from please_hold_still.masking import (
    VJEPA21_MASKS,
    BlockMaskConfig,
    ForegroundBlockMasker,
    equalize,
    sample_one,
    token_foreground,
)


def head_mask(b=2, t=16, size=256, radius=90):
    """(B, T, H, W) disk-shaped 'head' in every slice."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < radius**2
    return disk.expand(b, t, size, size).clone()


def test_token_foreground_shape_and_values():
    m = torch.zeros(1, 16, 256, 256, dtype=torch.bool)
    m[:, :, :16, :16] = True  # exactly the first token of every time step
    m[:, :2, 16:24, :16] = True  # half of token (t=0, h=1, w=0)
    fg = token_foreground(m)
    assert fg.shape == (1, 8, 16, 16)
    assert fg[0, :, 0, 0].eq(1).all() and fg[0, 0, 1, 0] == 0.5 and fg.sum() == 8.5


def test_targets_and_context_stay_on_anatomy_and_do_not_overlap():
    fg = token_foreground(head_mask(b=1))[0] > 0
    fg_idx = set(fg.flatten().nonzero().squeeze(1).tolist())
    g = torch.Generator().manual_seed(0)
    for cfg in VJEPA21_MASKS:
        for _ in range(20):
            ctx, tgt = sample_one(fg, cfg, g)
            c, t = set(ctx.tolist()), set(tgt.tolist())
            assert c and t and not (c & t)
            assert c | t <= fg_idx
            assert ctx.tolist() == sorted(c) and tgt.tolist() == sorted(t)


def test_blocks_run_through_all_slices():
    fg = torch.ones(8, 16, 16, dtype=torch.bool)
    _, tgt = sample_one(fg, BlockMaskConfig(1, 0.15), torch.Generator().manual_seed(1))
    hidden = torch.zeros(8 * 16 * 16, dtype=torch.bool)
    hidden[tgt] = True
    hidden = hidden.view(8, 16, 16)
    # temporal_scale=1.0: a hidden (row, col) is hidden at every time step.
    assert torch.equal(hidden.any(0), hidden.all(0))


def test_masking_ratios_are_in_the_vjepa_range():
    fg = token_foreground(head_mask(b=1))[0] > 0
    g = torch.Generator().manual_seed(2)
    short, long_ = [], []
    for _ in range(30):
        short.append(len(sample_one(fg, VJEPA21_MASKS[0], g)[1]) / fg.sum().item())
        long_.append(len(sample_one(fg, VJEPA21_MASKS[1], g)[1]) / fg.sum().item())
    assert 0.3 < sum(short) / 30 < 0.95
    assert 0.5 < sum(long_) / 30 < 0.99


def test_background_is_kept_in_context_when_asked():
    fg = token_foreground(head_mask(b=1))[0] > 0
    air = set((~fg).flatten().nonzero().squeeze(1).tolist())
    ctx, tgt = sample_one(
        fg, VJEPA21_MASKS[0], torch.Generator().manual_seed(3), drop_background=False
    )
    assert set(ctx.tolist()) & air  # visible air is shown to the student...
    assert not set(tgt.tolist()) & air  # ...but air is never a prediction target


def test_all_background_clip_falls_back_to_all_tokens():
    fg = torch.zeros(8, 16, 16, dtype=torch.bool)
    ctx, tgt = sample_one(fg, VJEPA21_MASKS[1], torch.Generator().manual_seed(4))
    assert len(ctx) > 0 and len(tgt) > 0


def test_single_anatomy_token_still_gets_targets():
    # Crashed run2 at step ~745: one such clip emptied the targets of the whole batch.
    fg = torch.zeros(8, 16, 16, dtype=torch.bool)
    fg[3, 7, 7] = True
    for cfg in VJEPA21_MASKS:
        ctx, tgt = sample_one(fg, cfg, torch.Generator().manual_seed(6))
        assert len(ctx) > 0 and len(tgt) > 0 and not set(ctx.tolist()) & set(tgt.tolist())


def test_batch_with_one_nearly_empty_clip_keeps_targets():
    token_fg = token_foreground(head_mask(b=3))
    token_fg[1] = 0
    token_fg[1, 3, 7, 7] = 1.0  # clip 1: anatomy in a single token
    masks_enc, masks_pred = ForegroundBlockMasker()(token_fg, torch.Generator().manual_seed(7))
    for m_enc, m_pred in zip(masks_enc, masks_pred, strict=True):
        assert m_enc.shape[1] > 0 and m_pred.shape[1] > 0


def test_tiny_head_still_gets_context():
    fg = torch.zeros(8, 16, 16, dtype=torch.bool)
    fg[:, 7:9, 7:9] = True  # 2x2 tokens: any big block swallows it
    ctx, tgt = sample_one(fg, BlockMaskConfig(2, 0.99), torch.Generator().manual_seed(5))
    assert len(ctx) > 0 and len(tgt) > 0 and not set(ctx.tolist()) & set(tgt.tolist())


def test_equalize_thins_randomly_to_shortest():
    lists = [torch.arange(10), torch.arange(100, 104), torch.arange(50, 57)]
    out = equalize(lists, torch.Generator().manual_seed(6))
    assert out.shape == (3, 4)
    assert torch.equal(out[1], torch.arange(100, 104))
    assert set(out[0].tolist()) <= set(range(10)) and out[0].tolist() == sorted(out[0].tolist())


def test_batch_output_format_and_reproducibility():
    mask = head_mask(b=3)
    mask[1, :, :, :128] = False  # a sample with a smaller head section
    masker = ForegroundBlockMasker()
    enc_a, pred_a = masker(token_foreground(mask), torch.Generator().manual_seed(7))
    enc_b, pred_b = masker(token_foreground(mask), torch.Generator().manual_seed(7))
    assert len(enc_a) == len(pred_a) == 2  # short-range and long-range
    for e, p in zip(enc_a, pred_a, strict=True):
        assert e.dtype == p.dtype == torch.long
        assert e.shape[0] == p.shape[0] == 3 and e.shape[1] > 0 and p.shape[1] > 0
        assert int(e.max()) < 2048 and int(p.max()) < 2048
    assert all(torch.equal(a, b) for a, b in zip(enc_a + pred_a, enc_b + pred_b, strict=True))
