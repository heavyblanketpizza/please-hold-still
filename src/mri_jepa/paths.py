"""Where data lives on disk.

Everything large (raw downloads, preprocessed arrays, model checkpoints, the
Hugging Face cache) goes under one *data root* on the external SSD, never on
the Mac's internal disk. Override the location with the MRI_JEPA_DATA
environment variable, e.g. for tests or another machine.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "MRI_JEPA_DATA"
DEFAULT_DATA_ROOT = Path("/Volumes/Just for Fun/mri-jepa-data")


def data_root() -> Path:
    """The data root: $MRI_JEPA_DATA if set, else the folder on the SSD."""
    return Path(os.environ.get(ENV_VAR, DEFAULT_DATA_ROOT)).expanduser()


def ensure_data_root(root: Path | None = None) -> Path:
    """Create the data root if needed and return it.

    Refuses to create the *parent* folder. If the SSD is not plugged in,
    /Volumes/Just for Fun does not exist, and silently creating it would put
    gigabytes of data on the internal disk instead.
    """
    root = Path(root) if root is not None else data_root()
    if not root.parent.is_dir():
        raise FileNotFoundError(
            f"{root.parent} does not exist. Is the external SSD plugged in? "
            f"(Or set {ENV_VAR} to another folder.)"
        )
    root.mkdir(exist_ok=True)
    return root


def raw_dir(root: Path | None = None) -> Path:
    """Downloaded OpenMind files (CSV + NIfTI volumes and masks)."""
    return (Path(root) if root is not None else data_root()) / "raw"


def processed_dir(root: Path | None = None) -> Path:
    """Preprocessed .npy volumes + JSON sidecars."""
    return (Path(root) if root is not None else data_root()) / "processed"


def hf_cache(root: Path | None = None) -> Path:
    """Hugging Face download caches (redirected here so they do not fill the internal disk).

    Only the caches move. The login token stays in ~/.cache/huggingface, where
    `hf auth login` puts it.
    """
    return (Path(root) if root is not None else data_root()) / "hf_cache"


def torch_home(root: Path | None = None) -> Path:
    """torch.hub cache: V-JEPA code and checkpoints."""
    return (Path(root) if root is not None else data_root()) / "torch_home"
