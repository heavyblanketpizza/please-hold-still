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
    "slice_height": "Where in the head",
    "sex": "Sex (M/F)",
    "age": "Age (years)",
    "modality": "Scan type (T1w/T2w/FLAIR)",
}
METRIC_LABELS = {"balanced_accuracy": "balanced accuracy ↑", "mae": "mean abs. error ↓"}


def plot_probe_comparison(results: dict[str, dict], out_path: str | Path) -> Path:
    """One small panel per probe task; one bar per model, with its 95% range.

    results: {model name: {task: probe result dict}} as returned by
    `please_hold_still.probe.run_probes`. Tasks skipped for any model are left out.
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


def plot_paired_changes(
    results: dict[str, dict], paired: dict[str, dict], out_path: str | Path
) -> Path:
    """One row per probe task: how each model's score differs from the first model's.

    results: {model name: {task: probe result dict}}, the first model is the
    reference. paired: {model name: `please_hold_still.probe.paired_differences`}
    for every model after the first. Each change is drawn as a percentage of
    the reference's score and turned so that right is always better (less
    error, or higher accuracy), so tasks with different units share one axis.
    The line is the paired 95% range. A filled dot means that range excludes
    0; a hollow dot means the change could be luck.
    """
    from matplotlib.lines import Line2D

    reference, *others = list(results)
    if not 1 <= len(others) < len(MODEL_COLORS):
        raise ValueError(f"need 2 to {len(MODEL_COLORS)} models")
    tasks = [
        t
        for t in TASK_TITLES
        if "value" in results[reference].get(t, {}) and all(t in paired.get(m, {}) for m in others)
    ]
    if not tasks:
        raise ValueError("no probe task has a paired comparison for every model")

    fig, ax = plt.subplots(figsize=(9, 0.9 * len(tasks) + 1.5), facecolor="white")
    offsets = np.linspace(-0.13, 0.13, len(others)) if len(others) > 1 else [0.0]
    widest = 0.0
    for row, task in enumerate(tasks):
        ref = results[reference][task]
        sign = 1 if ref["higher_is_better"] else -1
        for i, (model, dy) in enumerate(zip(others, offsets, strict=True)):
            d = paired[model][task]
            pct = 100 * sign * d["delta"] / ref["value"]
            lo, hi = sorted(100 * sign * v / ref["value"] for v in d["ci95"])
            widest = max(widest, abs(lo), abs(hi))
            clear = lo > 0 or hi < 0
            color = MODEL_COLORS[i + 1]  # same colour per model as in plot_probe_comparison
            ax.plot([lo, hi], [row + dy] * 2, color=color, linewidth=2, solid_capstyle="round")
            ax.plot(
                pct,
                row + dy,
                "o",
                markersize=8,
                markeredgewidth=2,
                color=color,
                markerfacecolor=color if clear else "white",
                zorder=3,
            )
            if clear:  # label only the changes that are not luck
                ax.text(
                    hi,
                    row + dy,
                    f"  {model} {pct:+.0f}%",
                    va="center",
                    fontsize=8,
                    color=TEXT_PRIMARY,
                )
        value = ref["value"]
        metric = "error" if ref["metric"] == "mae" else "balanced acc."
        label = ax.get_yaxis_transform()  # x in axes fractions, y in rows
        ax.text(
            -0.02,
            row - 0.1,
            TASK_TITLES[task],
            transform=label,
            ha="right",
            va="center",
            fontsize=10,
            color=TEXT_PRIMARY,
        )
        ax.text(
            -0.02,
            row + 0.2,
            f"{reference}: {metric} {value:.{3 if value < 1 else 2}f}",
            transform=label,
            ha="right",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )

    limit = 1.45 * widest if widest > 0 else 1.0  # room on the right for the labels
    ax.set_xlim(-limit, limit)
    ax.set_ylim(len(tasks) - 0.5, -0.5)  # first task at the top
    ax.set_yticks([])
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:+.0f}%".replace("-", "−") if round(x) else "0")
    ax.set_xlabel(f"change from {reference}, as % of its score", fontsize=9, color=TEXT_SECONDARY)
    for x, text, ha in ((0, "← worse", "left"), (1, "better →", "right")):
        ax.text(
            x,
            1.02,
            text,
            transform=ax.transAxes,
            ha=ha,
            va="bottom",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d4d3cc")
    ax.xaxis.grid(True, color="#eeede8", linewidth=0.8)
    ax.set_axisbelow(True)

    dot = {"marker": "o", "markersize": 7, "markeredgewidth": 2, "linewidth": 0}
    handles = [
        Line2D([], [], color=MODEL_COLORS[i + 1], linewidth=2, marker="o", markersize=7, label=m)
        for i, m in enumerate(others)
    ] + [
        Line2D([], [], color=TEXT_MUTED, label="clear change", **dot),
        Line2D([], [], color=TEXT_MUTED, markerfacecolor="white", label="could be luck", **dot),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    fig.text(
        0.5,
        0.015,
        "Dot: change on the same test scans · line: paired 95% range over resampled test volumes",
        ha="center",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))  # also makes room for the row labels

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    return out_path
