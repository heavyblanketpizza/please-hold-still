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


# Categorical slots 1-4 (light surface), assigned to models in the order given.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT_MUTED = "#8a897f"

TASK_TITLES = {
    "modality": "Scan type (T1w/T2w/FLAIR)",
    "sex": "Sex",
    "age": "Age (years)",
    "slice_height": "Where in the head",
}
METRIC_LABELS = {"balanced_accuracy": "balanced accuracy ↑", "mae": "mean abs. error ↓"}


def plot_probe_comparison(results: dict[str, dict], out_path: str | Path) -> Path:
    """One small panel per probe task; one bar per model, with its 95% range.

    results: {model name: {task: probe result dict}} as returned by
    `mri_jepa.probe.run_probes`. Tasks skipped for any model are left out.
    """
    from matplotlib.patches import Patch

    models = list(results)
    if len(models) > len(MODEL_COLORS):
        raise ValueError(f"at most {len(MODEL_COLORS)} models per chart")
    tasks = [t for t in TASK_TITLES if all("value" in results[m].get(t, {}) for m in models)]
    if not tasks:
        raise ValueError("no probe task has results for every model")

    fig, axes = plt.subplots(
        1, len(tasks), figsize=(3.1 * len(tasks), 3.6), facecolor="white", squeeze=False
    )
    for ax, task in zip(axes[0], tasks, strict=True):
        entries = [results[m][task] for m in models]
        values = np.array([e["value"] for e in entries])
        lo = values - np.array([e["ci95"][0] for e in entries])
        hi = np.array([e["ci95"][1] for e in entries]) - values
        x = np.arange(len(models))
        ax.bar(
            x, values, width=0.62, color=MODEL_COLORS[: len(models)], edgecolor="white", linewidth=2
        )
        ax.errorbar(
            x,
            values,
            yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)],
            fmt="none",
            ecolor=TEXT_SECONDARY,
            elinewidth=1,
            capsize=3,
        )
        chance = entries[0]["chance"]
        ax.axhline(chance, color=TEXT_MUTED, linestyle="--", linewidth=1)
        ax.text(
            len(models) - 0.5,
            chance,
            " chance",
            va="bottom",
            ha="right",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
        top = max(float(np.max(values + np.clip(hi, 0, None))), chance) * 1.18
        for xi, v in zip(x, values, strict=True):
            ax.text(
                xi,
                v / 2 if v > top * 0.12 else v,
                f"{v:.2f}",
                ha="center",
                va="center" if v > top * 0.12 else "bottom",
                fontsize=9,
                color="white" if v > top * 0.12 else TEXT_PRIMARY,
            )
        ax.set_ylim(0, top)
        ax.set_xticks([])
        ax.set_title(TASK_TITLES[task], fontsize=10, color=TEXT_PRIMARY, pad=16)
        ax.text(
            0.5,
            1.01,
            METRIC_LABELS[entries[0]["metric"]],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d4d3cc")
        ax.yaxis.grid(True, color="#eeede8", linewidth=0.8)
        ax.set_axisbelow(True)

    handles = [Patch(color=MODEL_COLORS[i], label=m) for i, m in enumerate(models)]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(models),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    fig.text(
        0.5,
        0.015,
        "Bars: test-set score · whiskers: 95% range over resampled test volumes · dashed: chance",
        ha="center",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    return out_path
