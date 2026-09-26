"""Frozen-encoder probes: how much useful information is in the encoder's features?

1. `extract_features`: for every volume, take several clips spread from the
   bottom to the top of the head. Run the frozen encoder and average its
   tokens over the anatomy (air is ignored). Result: one 768-number vector
   per clip.
2. `attach_metadata_labels`: add age and sex from openneuro_metadata.csv,
   where known.
3. `run_probes`: fit small linear models on the *train* studies and score
   them on the held-out *test* studies:
     modality      T1w / T2w / FLAIR, per volume  -> balanced accuracy
     sex           per volume, if labelled        -> balanced accuracy
     age           per volume, if labelled        -> mean absolute error (years)
     slice_height  where in the head a clip sits  -> mean absolute error
                   (0 = bottom of the head, 1 = top)
   Each score comes with a chance-level reference and a bootstrap 95% range
   (resampling test volumes), so small differences can be told from noise.

Run it on the original encoder ("before") and on a trained one ("after"),
with the same data and split, then compare.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from please_hold_still.data.dataset import MRISliceClipDataset, find_volumes
from please_hold_still.masking import token_foreground

MIN_TRAIN, MIN_TEST = 10, 4


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def clip_starts(depth: int, num_frames: int, n: int) -> list[int]:
    """`n` clip start slices spread evenly from the bottom to the top of a volume."""
    last = max(depth - num_frames, 0)
    return sorted({int(round(s)) for s in np.linspace(0, last, n)})


def pool_tokens(tokens: torch.Tensor, token_fg: torch.Tensor) -> torch.Tensor:
    """Average tokens (B, N, D) over anatomy tokens, token_fg (B, t, h, w) -> (B, D)."""
    w = (token_fg.flatten(1) > 0).float()
    w[w.sum(1) == 0] = 1.0  # a clip with no anatomy at all: use every token
    return (tokens * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True)


@torch.inference_mode()
def extract_features(
    encoder: torch.nn.Module,
    processed_dir,
    clips_per_volume: int = 8,
    num_frames: int = 16,
    size: int = 256,
    batch_size: int = 8,
    progress: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """One pooled feature vector per clip, plus a table describing each clip."""
    from tqdm import tqdm

    device = next(encoder.parameters()).device
    ds = MRISliceClipDataset(processed_dir, num_frames=num_frames, size=size, train=False)
    jobs = [
        (i, s)
        for i, v in enumerate(ds.items)
        for s in clip_starts(v["shape"][0], num_frames, clips_per_volume)
    ]
    feats, rows = [], []
    t0 = time.time()
    for b in tqdm(range(0, len(jobs), batch_size), desc="features", disable=not progress):
        items = [ds.clip(i, start=s) for i, s in jobs[b : b + batch_size]]
        video = torch.stack([it["video"] for it in items]).permute(0, 2, 1, 3, 4)
        mask = torch.stack([it["mask"] for it in items])
        tokens = encoder(video.to(device))
        tubelet, patch = encoder.tubelet_size, encoder.patch_size
        fg = token_foreground(mask, tubelet, patch).to(device)
        feats.append(pool_tokens(tokens.float(), fg).cpu().numpy())
        for (i, _), it in zip(jobs[b : b + batch_size], items, strict=True):
            v = ds.items[i]
            rows.append(
                {
                    "id": v["id"],
                    "dataset_id": v["source_dataset"],
                    "split": v["split"],
                    "modality": v["modality"],
                    "source_image": v["source_image"],
                    "start": it["start"],
                    "depth": it["depth"],
                    "slice_height": min(1.0, (it["start"] + num_frames / 2) / it["depth"]),
                }
            )
        if progress and b == 0 and len(jobs) > batch_size:
            per_batch = time.time() - t0
            eta_min = per_batch * (len(jobs) / batch_size) / 60
            tqdm.write(f"{len(jobs)} clips from {len(ds)} volumes; about {eta_min:.0f} min")
    return np.concatenate(feats).astype(np.float32), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Labels from the OpenMind metadata
# ---------------------------------------------------------------------------


def _normalise_sex(value) -> str | float:
    if isinstance(value, str) and value.strip():
        first = value.strip().lower()[0]
        if first in ("m", "f"):
            return first.upper()
    return np.nan


def attach_metadata_labels(clips: pd.DataFrame, metadata_csv) -> pd.DataFrame:
    """Add `age` (years, float) and `sex` ("M"/"F") columns where the CSV has them."""
    meta = pd.read_csv(metadata_csv, low_memory=False)
    keep = ["image_path"] + [c for c in ("age", "sex") if c in meta.columns]
    meta = meta[keep].drop_duplicates("image_path")
    # sidecars store the repo path, e.g. "OpenMind/ds000117/...": strip the folder prefix
    rel = clips["source_image"].str.replace(r"^OpenMind/", "", regex=True)
    out = clips.assign(image_path=rel).merge(meta, on="image_path", how="left")
    out["age"] = pd.to_numeric(out["age"], errors="coerce") if "age" in out else np.nan
    out["sex"] = out["sex"].map(_normalise_sex) if "sex" in out else np.nan
    return out.drop(columns="image_path")


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _bootstrap_ci(metric, y_true, y_pred, groups, n=1000, seed=0) -> list[float]:
    """95% range of `metric` when test volumes are resampled with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    values = []
    with warnings.catch_warnings():  # a resample can miss a class; that is expected
        warnings.simplefilter("ignore")
        for _ in range(n):
            pick = np.concatenate([index[g] for g in rng.choice(uniq, len(uniq))])
            try:
                values.append(metric(y_true[pick], y_pred[pick]))
            except ValueError:
                continue
    lo, hi = np.percentile(values, [2.5, 97.5]) if values else (np.nan, np.nan)
    return [float(lo), float(hi)]


def _classifier(y_train: np.ndarray):
    folds = int(min(5, pd.Series(y_train).value_counts().min()))
    if folds >= 2:
        clf = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 6),
            cv=folds,
            max_iter=5000,
            scoring="balanced_accuracy",
            l1_ratios=(0.0,),
            use_legacy_attributes=False,
        )
    else:
        clf = LogisticRegression(max_iter=5000)
    return make_pipeline(StandardScaler(), clf)


def _regressor():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15)))


def _probe(kind, x_tr, y_tr, x_te, y_te, groups_te) -> dict:
    if len(y_tr) < MIN_TRAIN or len(y_te) < MIN_TEST:
        return {"skipped": f"too few labelled samples (train {len(y_tr)}, test {len(y_te)})"}
    if kind == "classification":
        classes = np.unique(y_tr)
        if len(classes) < 2 or len(np.unique(y_te)) < 2:
            return {"skipped": "needs at least two classes in train and in test"}
        pred = _classifier(y_tr).fit(x_tr, y_tr).predict(x_te)
        return {
            "metric": "balanced_accuracy",
            "higher_is_better": True,
            "value": float(balanced_accuracy_score(y_te, pred)),
            "ci95": _bootstrap_ci(balanced_accuracy_score, y_te, pred, groups_te),
            "chance": 1.0 / len(classes),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
        }
    pred = _regressor().fit(x_tr, y_tr).predict(x_te)
    baseline = np.full_like(y_te, y_tr.mean(), dtype=float)
    return {
        "metric": "mae",
        "higher_is_better": False,
        "value": float(mean_absolute_error(y_te, pred)),
        "ci95": _bootstrap_ci(mean_absolute_error, y_te, pred, groups_te),
        "chance": float(mean_absolute_error(y_te, baseline)),  # always guess the train mean
        "r2": float(r2_score(y_te, pred)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }


TASKS = {
    # name: (label column, per "volume" or per "clip", kind)
    "modality": ("modality", "volume", "classification"),
    "sex": ("sex", "volume", "classification"),
    "age": ("age", "volume", "regression"),
    "slice_height": ("slice_height", "clip", "regression"),
}


def run_probes(features: np.ndarray, clips: pd.DataFrame) -> dict[str, dict]:
    """Fit each probe on the train studies, score it on the test studies."""
    clips = clips.reset_index(drop=True)
    # Volume-level features: the average of that volume's clip features.
    vol_ids = clips["id"].to_numpy()
    uniq = list(dict.fromkeys(vol_ids))
    vol_feats = np.stack([features[vol_ids == v].mean(0) for v in uniq])
    volumes = clips.drop_duplicates("id").set_index("id").loc[uniq].reset_index()

    results = {}
    for task, (column, level, kind) in TASKS.items():
        table, x = (volumes, vol_feats) if level == "volume" else (clips, features)
        if column not in table:
            results[task] = {"skipped": f"no '{column}' column"}
            continue
        known = table[column].notna().to_numpy()
        train = known & (table["split"] == "train").to_numpy()
        test = known & (table["split"] == "test").to_numpy()
        y = table[column].to_numpy()
        if kind == "regression":
            y = y.astype(float)
        results[task] = _probe(
            kind, x[train], y[train], x[test], y[test], table["id"].to_numpy()[test]
        )
    return results


# ---------------------------------------------------------------------------
# Comparing models
# ---------------------------------------------------------------------------


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Rows = tasks; one column per model (score [95% range]) and the change first -> last.

    "within noise" means the two 95% ranges overlap: the difference could be luck.
    """
    models = list(results)
    rows = []
    for task in TASKS:
        entries = [results[m].get(task, {}) for m in models]
        if any("value" not in e for e in entries):
            continue
        higher = entries[0]["higher_is_better"]
        row = {
            "task": task,
            "metric": entries[0]["metric"] + (" (higher=better)" if higher else " (lower=better)"),
            "chance": round(entries[0]["chance"], 3),
        }
        for m, e in zip(models, entries, strict=True):
            row[m] = f"{e['value']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"
        if len(models) >= 2:
            first, last = entries[0], entries[-1]
            delta = last["value"] - first["value"]
            better = delta > 0 if higher else delta < 0
            overlap = last["ci95"][0] <= first["ci95"][1] and first["ci95"][0] <= last["ci95"][1]
            verdict = "better" if better else "worse" if delta != 0 else "same"
            row["change"] = f"{delta:+.3f} ({verdict}{', within noise' if overlap else ''})"
        rows.append(row)
    return pd.DataFrame(rows)


def stale_features_message(clips: pd.DataFrame, processed_dir) -> str | None:
    """None if saved features cover exactly the volumes preprocessed now, else what changed.

    After a bigger download, old features would silently score a model on the
    old, smaller set of volumes.
    """
    saved = set(clips["id"])
    current = {v["id"] for v in find_volumes(processed_dir)}
    if saved == current:
        return None
    return (
        f"these features cover {len(saved)} volumes, but {len(current)} are preprocessed "
        f"now ({len(current - saved)} new, {len(saved - current)} gone)"
    )


def mismatched_volumes_message(summaries: dict[str, dict]) -> str | None:
    """None if every results.json was scored on the same number of volumes, else a listing.

    Scores from different test sets cannot be compared fairly.
    """
    counts = {tag: (s["n_volumes"], s["n_test_volumes"]) for tag, s in summaries.items()}
    if len(set(counts.values())) <= 1:
        return None
    lines = [f"  {tag}: {n} volumes, {t} of them for testing" for tag, (n, t) in counts.items()]
    return "these models were scored on different volumes:\n" + "\n".join(lines)
