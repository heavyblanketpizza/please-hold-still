"""V-JEPA loader helper.

The tests that build the real architecture need a local clone of
facebookresearch/vjepa2 and are skipped unless VJEPA2_REPO points at one:

    git clone https://github.com/facebookresearch/vjepa2 /some/folder
    VJEPA2_REPO=/some/folder uv run pytest tests/test_vjepa.py
"""

import os

import pytest
import torch

from please_hold_still.models import vjepa

VJEPA2_REPO = os.environ.get("VJEPA2_REPO")
needs_repo = pytest.mark.skipif(not VJEPA2_REPO, reason="set VJEPA2_REPO to a vjepa2 clone")


def test_num_tokens():
    # 16 frames -> 8 time steps; 256 px -> 16 patches per side.
    assert vjepa.num_tokens(16, 256, 256) == 8 * 16 * 16 == 2048
    assert vjepa.num_tokens(16, 384, 384) == 8 * 24 * 24


def test_clean_state_dict():
    state = {"module.backbone.blocks.0.w": 1, "module.norm.b": 2, "pos": 3}
    assert vjepa.clean_state_dict(state) == {"blocks.0.w": 1, "norm.b": 2, "pos": 3}


def test_pick_device():
    assert vjepa.pick_device("cpu") == torch.device("cpu")
    assert vjepa.pick_device().type in {"cpu", "mps", "cuda"}


@needs_repo
def test_encoder_output_shape_random_weights():
    encoder = vjepa.load_vjepa2_1_encoder(pretrained=False, device="cpu", hub_repo=VJEPA2_REPO)
    assert sum(p.numel() for p in encoder.parameters()) > 80e6  # ViT-B, ~87M
    with torch.inference_mode():
        out = encoder(torch.zeros(1, 3, 4, 64, 64))  # small clip keeps the test fast
    assert tuple(out.shape) == (1, vjepa.num_tokens(4, 64, 64), 768)


@needs_repo
def test_checkpoint_loading_round_trip(tmp_path):
    """Save weights the way Meta's checkpoints store them, then load them back."""
    src_enc, src_pred = vjepa.build_architecture("vit_base", VJEPA2_REPO)

    def wrap(model):
        return {f"module.backbone.{k}": v for k, v in model.state_dict().items()}

    ckpt = tmp_path / "fake.pt"
    torch.save({"ema_encoder": wrap(src_enc), "predictor": wrap(src_pred), "epoch": 1}, ckpt)

    enc, pred = vjepa.load_vjepa2_1(
        checkpoint=ckpt, device="cpu", hub_repo=VJEPA2_REPO, with_predictor=True
    )
    for a, b in zip(src_enc.state_dict().values(), enc.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    for a, b in zip(src_pred.state_dict().values(), pred.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert not enc.training  # returned in eval mode


@needs_repo
def test_explicit_checkpoint_is_loaded_even_without_pretrained(tmp_path):
    enc, _ = vjepa.build_architecture("vit_base", VJEPA2_REPO)
    with torch.no_grad():
        for p in enc.parameters():
            p.fill_(0.5)
    torch.save({"ema_encoder": enc.state_dict()}, tmp_path / "e.pt")
    loaded = vjepa.load_vjepa2_1_encoder(
        pretrained=False, checkpoint=tmp_path / "e.pt", device="cpu", hub_repo=VJEPA2_REPO
    )
    assert all(torch.all(p == 0.5) for p in loaded.parameters())
