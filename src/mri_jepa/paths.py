"""Where data lives on disk.

Everything large (raw downloads, preprocessed arrays, model checkpoints, the
Hugging Face cache) goes under one *data root*, ideally on an external drive.
Its location comes from the MRI_JEPA_DATA environment variable (or a script's
--data-root flag). There is deliberately no built-in default: machine-specific
paths stay out of the repository, and a missing setting stops the scripts
instead of silently filling the internal disk.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "MRI_JEPA_DATA"


def data_root() -> Path:
    """The data root from $MRI_JEPA_DATA. Raises a helpful error if it is not set."""
    value = os.environ.get(ENV_VAR, "").strip()
    if not value:
        raise RuntimeError(
            f"{ENV_VAR} is not set. Add a line like this to ~/.zshrc, then open a "
            f'new terminal:\n    export {ENV_VAR}="/path/to/your/data-folder"\n'
            "(or pass --data-root to the script)."
        )
    return Path(value).expanduser()


def ensure_data_root(root: Path | None = None) -> Path:
    """Create the data root if needed and return it.

    Refuses to create the *parent* folder. If an external drive is not plugged
    in, its /Volumes/<name> folder does not exist, and silently creating it
    would put gigabytes of data on the internal disk instead.
    """
    root = Path(root) if root is not None else data_root()
    if not root.parent.is_dir():
        raise FileNotFoundError(
            f"{root.parent} does not exist. Is the external drive plugged in? "
            f"(Or point {ENV_VAR} at another folder.)"
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


def runs_dir(root: Path | None = None) -> Path:
    """Training runs: checkpoints and logs, one folder per run."""
    return (Path(root) if root is not None else data_root()) / "runs"


def eval_dir(root: Path | None = None) -> Path:
    """Probe evaluations: features and results, one folder per evaluated model."""
    return (Path(root) if root is not None else data_root()) / "eval"
