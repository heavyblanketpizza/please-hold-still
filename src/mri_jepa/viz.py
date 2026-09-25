"""Pictures of clips: a grid of axial slices with the anatomy mask overlaid."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

MASK_COLOR = "#3987e5"  # categorical slot 1 (blue), dark-surface step: sits on black MRI air
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e57"


def _to_numpy(a) -> np.ndarray:
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def plot_clip_grid(
    video,
    mask=None,
    out_path: str | Path = "clip.png",
    title: str = "",
    first_slice: int = 0,
    ncols: int = 4,
    nrows: int = 4,
    alpha: float = 0.18,
) -> Path:
    """Save a grid of the first `nrows * ncols` frames of a clip, mask overlaid.

    video: (T, 3, H, W) or (T, H, W), values in [-1, 1] (as the Dataset returns).
    mask:  (T, H, W) bool, or None for no overlay.
    Frames are drawn with origin="lower", so anterior is up and the patient's
    right is on the right (the stored RAS orientation, not flipped).
    """
    video = _to_numpy(video)
    grey = video[:, 0] if video.ndim == 4 else video
    mask = _to_numpy(mask).astype(bool) if mask is not None else None
    n = min(len(grey), nrows * ncols)

    fill = ListedColormap([MASK_COLOR])
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.4 * ncols, 2.7 * nrows + 0.7),
        facecolor="white",
        gridspec_kw={"hspace": 0.16, "wspace": 0.04},  # row gap leaves room for slice labels
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.75 / (2.7 * nrows + 0.7))
    for k, ax in enumerate(np.ravel(axes)):
        ax.set_axis_off()
        if k >= n:
            continue
        ax.imshow(grey[k], cmap="gray", vmin=-1, vmax=1, origin="lower", interpolation="nearest")
        if mask is not None and mask[k].any():
            ax.imshow(
                np.ma.masked_where(~mask[k], mask[k]),
                cmap=fill,
                alpha=alpha,
                origin="lower",
                interpolation="nearest",
            )
            ax.contour(mask[k], levels=[0.5], colors=[MASK_COLOR], linewidths=0.8, origin="lower")
        ax.set_title(f"z = {first_slice + k}", fontsize=9, color=TEXT_SECONDARY, pad=3)

    subtitle = "anatomy mask in blue · anterior ↑ · patient right →" if mask is not None else ""
    height = 2.7 * nrows + 0.7  # inches; place the two title lines in the top margin
    fig.suptitle(title, fontsize=12, color=TEXT_PRIMARY, y=1 - 0.12 / height, va="top")
    fig.text(
        0.5, 1 - 0.42 / height, subtitle, ha="center", va="top", fontsize=9, color=TEXT_SECONDARY
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)
    return out_path
