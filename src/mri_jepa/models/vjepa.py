"""Load Meta's V-JEPA 2.1 encoder (and optionally its predictor) via torch.hub.

Why not just `torch.hub.load(..., pretrained=True)`? At the time of writing,
Meta's hub code points its download URL at a test server
(`VJEPA_BASE_URL = "http://localhost:8300"` in src/hub/backbones.py), so that
fails. Instead we:

1. build the model *architecture* with torch.hub, `pretrained=False`, from a
   pinned commit of facebookresearch/vjepa2 (so later upstream changes cannot
   silently change our model);
2. download the checkpoint ourselves from Meta's real server into
   <data root>/torch_home/checkpoints (once, with a size check);
3. load the weights into the model.

The V-JEPA 2.1 models were trained on 384x384 frames. They use RoPE
(rotary) position encoding, which adapts to other sizes, so 256x256 input
works. The encoder takes a video tensor (B, 3, T, H, W) and returns patch
tokens (B, N, D) with N = (T / 2) * (H / 16) * (W / 16): each token covers
2 frames x 16 x 16 pixels. ViT-B: D = 768.
"""

from __future__ import annotations

import pickle
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from mri_jepa import paths

VJEPA_REPO = "facebookresearch/vjepa2"
VJEPA_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"  # main on 2026-03-23
CHECKPOINT_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
GB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    hub_entry: str  # function name in vjepa2's hubconf.py
    checkpoint: str  # file name (without .pt) on Meta's server
    encoder_key: str  # which weights in the checkpoint to use for the encoder
    patch_size: int = 16
    tubelet_size: int = 2


MODELS = {
    "vit_base": ModelSpec("vjepa2_1_vit_base_384", "vjepa2_1_vitb_dist_vitG_384", "ema_encoder"),
    "vit_large": ModelSpec("vjepa2_1_vit_large_384", "vjepa2_1_vitl_dist_vitG_384", "ema_encoder"),
}


def pick_device(name: str | None = None) -> torch.device:
    """`name` if given, else Apple GPU (mps) if available, else CUDA, else CPU."""
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def num_tokens(num_frames: int, height: int, width: int, model: str = "vit_base") -> int:
    """How many tokens the encoder outputs for a clip of this size."""
    spec = MODELS[model]
    time_steps = num_frames // spec.tubelet_size
    return time_steps * (height // spec.patch_size) * (width // spec.patch_size)


def build_architecture(model: str = "vit_base", hub_repo: str | Path | None = None):
    """Build (encoder, predictor) with random weights via torch.hub.

    `hub_repo`: None downloads the pinned commit from GitHub (cached under
    <data root>/torch_home/hub). A local folder containing a clone of
    facebookresearch/vjepa2 is used as-is (handy offline).
    """
    spec = MODELS[model]
    if hub_repo is not None and Path(hub_repo).is_dir():
        repo, source = str(hub_repo), "local"
    else:
        repo, source = str(hub_repo or f"{VJEPA_REPO}:{VJEPA_COMMIT}"), "github"
        torch.hub.set_dir(str(paths.torch_home() / "hub"))
    with warnings.catch_warnings():  # Meta's code uses a deprecated timm import path
        warnings.simplefilter("ignore", FutureWarning)
        kwargs = {"trust_repo": True, "skip_validation": True} if source == "github" else {}
        encoder, predictor = torch.hub.load(
            repo, spec.hub_entry, source=source, pretrained=False, **kwargs
        )
    return encoder, predictor


def remote_size(url: str) -> int | None:
    """File size in bytes from an HTTP HEAD request (None if the server does not say)."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
        length = r.headers.get("Content-Length")
    return int(length) if length else None


def download_checkpoint(model: str = "vit_base", max_gb: float = 5.0) -> Path:
    """Download the checkpoint once into <data root>/torch_home/checkpoints and return its path."""
    spec = MODELS[model]
    dst = paths.torch_home() / "checkpoints" / f"{spec.checkpoint}.pt"
    if dst.is_file():
        return dst
    url = f"{CHECKPOINT_BASE_URL}/{spec.checkpoint}.pt"
    size = remote_size(url)
    print(f"checkpoint {url}: {size / GB:.2f} GB" if size else f"checkpoint {url}: size unknown")
    if size and size > max_gb * GB:
        raise RuntimeError(f"checkpoint is bigger than max_gb={max_gb}; raise it to proceed")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(url, str(dst), progress=True)  # temp file + rename
    return dst


def clean_state_dict(state: dict) -> dict:
    """Strip the "module." / "backbone." prefixes training wrappers add to weight names."""
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}


def read_checkpoint(path: str | Path) -> dict:
    """torch.load a checkpoint, preferring the safe tensors-only mode."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Some training checkpoints also store plain Python objects. This file
        # comes from Meta's official server, so the full loader is acceptable.
        return torch.load(path, map_location="cpu", weights_only=False)


def load_vjepa2_1(
    model: str = "vit_base",
    pretrained: bool = True,
    device: str | torch.device | None = None,
    hub_repo: str | Path | None = None,
    checkpoint: str | Path | None = None,
    with_predictor: bool = False,
    max_gb: float = 5.0,
):
    """Load a V-JEPA 2.1 encoder, in eval mode, on `device` (default: best available).

    - `pretrained=False` keeps random weights (for offline tests).
    - `checkpoint`: use this local .pt instead of downloading.
    - `with_predictor=True` returns (encoder, predictor). Continued JEPA
      pretraining needs both.
    """
    spec = MODELS[model]
    encoder, predictor = build_architecture(model, hub_repo)
    if pretrained:
        state = read_checkpoint(checkpoint or download_checkpoint(model, max_gb))
        encoder.load_state_dict(clean_state_dict(state[spec.encoder_key]), strict=True)
        if with_predictor:
            predictor.load_state_dict(clean_state_dict(state["predictor"]), strict=True)
        del state
    device = pick_device(device) if not isinstance(device, torch.device) else device
    encoder = encoder.to(device).eval()
    if with_predictor:
        return encoder, predictor.to(device).eval()
    return encoder


def load_vjepa2_1_encoder(model: str = "vit_base", **kwargs):
    """Just the encoder. See `load_vjepa2_1` for the options."""
    return load_vjepa2_1(model, with_predictor=False, **kwargs)
