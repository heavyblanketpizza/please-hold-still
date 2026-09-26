"""Frozen-encoder probes and the before/after comparison."""

import json

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from conftest import make_phantom
from torch import nn

from please_hold_still import probe
from please_hold_still.data import preprocess as pp
from please_hold_still.viz import plot_probe_comparison


def test_clip_starts():
    assert probe.clip_starts(100, 16, 5) == [0, 21, 42, 63, 84]
    assert probe.clip_starts(10, 16, 8) == [0]  # volume shorter than a clip


def test_pool_tokens_ignores_air():
    tokens = torch.tensor([[[1.0], [3.0], [100.0], [100.0]]])  # (1, 4 tokens, 1)
    fg = torch.tensor([[[[1.0, 0.5], [0.0, 0.0]]]])  # (1, t=1, h=2, w=2)
    assert probe.pool_tokens(tokens, fg).item() == 2.0
    all_air = torch.zeros_like(fg)
    assert probe.pool_tokens(tokens, all_air).item() == pytest.approx(51.0)


def test_attach_metadata_labels(tmp_path):
    csv = tmp_path / "meta.csv"
    pd.DataFrame(
        {
            "image_path": ["ds1/a.nii.gz", "ds1/b.nii.gz", "ds2/c.nii.gz"],
            "age": ["34", "n/a", 61],
            "sex": ["male", "F", "other"],
        }
    ).to_csv(csv, index=False)
    clips = pd.DataFrame(
        {"source_image": ["OpenMind/ds1/a.nii.gz", "OpenMind/ds1/b.nii.gz", "ds2/c.nii.gz"]}
    )
    out = probe.attach_metadata_labels(clips, csv)
    assert out["age"].tolist()[0] == 34 and np.isnan(out["age"][1]) and out["age"][2] == 61
    assert out["sex"].tolist()[:2] == ["M", "F"] and pd.isna(out["sex"][2])


def synthetic(n_volumes=80, clips_per=4, signal=1.0, seed=0):
    """Fake features: modality, age and slice height are linearly encoded (x signal)."""
    rng = np.random.default_rng(seed)
    rows, feats = [], []
    mods = ["T1w", "T2w", "FLAIR"]
    for v in range(n_volumes):
        mod, age = mods[v % 3], rng.uniform(20, 80)
        sex = rng.choice(["M", "F"])  # NOT encoded: should stay near chance
        for c in range(clips_per):
            height = (c + 0.5) / clips_per
            f = rng.normal(0, 1, 16)
            f[v % 3] += 3 * signal  # modality
            f[5] += signal * (age - 50) / 15
            f[6] += signal * 8 * (height - 0.5)
            feats.append(f)
            rows.append(
                {
                    "id": f"vol{v}",
                    "split": "test" if v % 5 == 0 else "train",
                    "modality": mod,
                    "age": age,
                    "sex": sex,
                    "slice_height": height,
                }
            )
    return np.array(feats, dtype=np.float32), pd.DataFrame(rows)


def test_probes_find_encoded_information_and_not_the_rest():
    feats, clips = synthetic()
    r = probe.run_probes(feats, clips)
    assert r["modality"]["value"] > 0.9 and r["modality"]["chance"] == pytest.approx(1 / 3)
    assert r["age"]["value"] < 0.6 * r["age"]["chance"]  # much better than guessing the mean
    assert r["slice_height"]["value"] < 0.6 * r["slice_height"]["chance"]
    assert r["sex"]["value"] < 0.8  # sex was not encoded
    lo, hi = r["modality"]["ci95"]
    assert lo <= r["modality"]["value"] <= hi
    assert r["modality"]["n_test"] == 16 and r["slice_height"]["n_test"] == 64


def test_probes_skip_tasks_without_labels():
    feats, clips = synthetic(n_volumes=30)
    r = probe.run_probes(feats, clips.drop(columns=["age"]).assign(sex=np.nan))
    assert "skipped" in r["age"] and "skipped" in r["sex"] and "value" in r["modality"]


def test_comparison_table_and_chart(tmp_path):
    weak = probe.run_probes(*synthetic(signal=0.2))
    strong = probe.run_probes(*synthetic(signal=1.5))
    table = probe.comparison_table({"baseline": weak, "trained": strong})
    columns = ["task", "metric", "chance", "baseline", "trained", "trained change"]
    assert list(table.columns) == columns
    row = table.set_index("task").loc["slice_height"]
    assert "better" in row["trained change"] and "within noise" not in row["trained change"]

    out = plot_probe_comparison({"baseline": weak, "trained": strong}, tmp_path / "cmp.png")
    assert mpimg.imread(out).shape[1] > 800


def test_predictions_reproduce_the_scores_after_a_csv_round_trip(tmp_path):
    results, preds = probe.run_probes(*synthetic(), return_predictions=True)
    preds.to_csv(tmp_path / "predictions.csv", index=False)
    preds = pd.read_csv(tmp_path / "predictions.csv")  # labels come back as text
    assert preds.groupby("task").size().to_dict() == {
        "age": 16,
        "modality": 16,
        "sex": 16,
        "slice_height": 64,
    }
    same = probe.paired_differences(preds, preds)
    assert all(d["delta"] == 0 and d["ci95"] == [0, 0] for d in same.values())
    mae = probe.METRICS["regression"]
    rows = preds[preds["task"] == "slice_height"]
    y_true, y_pred = rows["y_true"].astype(float), rows["y_pred"].astype(float)
    assert mae(y_true, y_pred) == pytest.approx(results["slice_height"]["value"])


def test_paired_comparison_sees_a_small_gain_that_overlapping_ranges_miss():
    # Both models make the same per-volume mistakes; model b's are 10% smaller.
    rng = np.random.default_rng(0)
    n_vol, n_clip = 40, 5
    ids = np.repeat([f"v{i}" for i in range(n_vol)], n_clip)
    y = rng.uniform(0, 1, n_vol * n_clip)
    shared = np.repeat(rng.normal(0, 0.3, n_vol), n_clip)
    a = pd.DataFrame(
        {
            "task": "slice_height",
            "id": ids,
            "clip": np.tile(np.arange(n_clip), n_vol),
            "y_true": y,
            "y_pred": y + shared,
        }
    )
    b = a.assign(y_pred=y + 0.9 * shared)

    mae = probe.METRICS["regression"]
    lo_a, hi_a = probe._bootstrap_ci(mae, y, a["y_pred"].to_numpy(), ids)
    lo_b, hi_b = probe._bootstrap_ci(mae, y, b["y_pred"].to_numpy(), ids)
    assert lo_a <= hi_b and lo_b <= hi_a  # the separate ranges overlap...

    diff = probe.paired_differences(a, b)["slice_height"]
    assert diff["delta"] == pytest.approx(mae(y, b["y_pred"]) - mae(y, a["y_pred"]))
    assert diff["ci95"][1] < 0  # ...but the paired range is clearly below 0

    table = probe.comparison_table(
        {
            "a": {"slice_height": _result(mae(y, a["y_pred"]), [lo_a, hi_a])},
            "b": {"slice_height": _result(mae(y, b["y_pred"]), [lo_b, hi_b])},
        },
        paired={"b": {"slice_height": diff}},
    )
    change = table.set_index("task").loc["slice_height", "b change"]
    assert "better" in change and "within noise" not in change and "[" in change


def _result(value, ci95):
    return {"metric": "mae", "higher_is_better": False, "value": value, "ci95": ci95, "chance": 1}


def test_paired_comparison_refuses_different_test_items():
    _, preds = probe.run_probes(*synthetic(), return_predictions=True)
    with pytest.raises(ValueError, match="same test items"):
        probe.paired_differences(preds, preds.iloc[1:])
    relabelled = preds.assign(y_true=preds["y_true"].where(preds["task"] != "modality", "T1w"))
    with pytest.raises(ValueError, match="different true labels"):
        probe.paired_differences(preds, relabelled)


class PatchMeanEncoder(nn.Module):
    """Stand-in for V-JEPA: one token per 2x16x16 block, features = block means."""

    patch_size, tubelet_size, embed_dim = 16, 2, 6

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))  # so .parameters() has a device

    def forward(self, x):
        p = F.avg_pool3d(x, (2, 16, 16))  # (B, 3, t, h, w)
        tok = p.flatten(2).transpose(1, 2)  # (B, N, 3)
        return torch.cat([tok, tok**2], dim=-1)


def _write_sidecars(folder, ids):
    folder.mkdir(parents=True, exist_ok=True)
    for i in ids:
        sidecar = {"id": i, "source_dataset": folder.name, "modality": "T1w"}
        (folder / f"{i}.json").write_text(json.dumps(sidecar))


def test_stale_features_after_more_data(tmp_path):
    _write_sidecars(tmp_path / "ds000001", ["a", "b"])
    clips = pd.DataFrame({"id": ["a", "a", "b", "b"]})  # several clips per volume
    assert probe.stale_features_message(clips, tmp_path) is None

    _write_sidecars(tmp_path / "ds000002", ["c"])  # a bigger download added a volume
    msg = probe.stale_features_message(clips, tmp_path)
    assert "cover 2 volumes" in msg and "3 are preprocessed" in msg and "1 new" in msg


def test_mismatched_volumes():
    small = {"n_volumes": 200, "n_test_volumes": 33}
    assert probe.mismatched_volumes_message({"baseline": small, "run1": small}) is None

    big = {"n_volumes": 2000, "n_test_volumes": 400}
    msg = probe.mismatched_volumes_message({"baseline": small, "run2": big})
    assert "baseline: 200 volumes" in msg and "run2: 2000 volumes" in msg


def test_extract_features_end_to_end(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    rows = []
    for i in range(2):
        f = make_phantom(raw, f"v{i}", seed=i)
        rows.append(
            {
                "id": f"ds00000{i}__v{i}",
                "dataset_id": f"ds00000{i}",
                "modality": "T1w",
                "image": f["image"].name,
                "anat_mask": f["anat"].name,
                "anon_mask": "",
            }
        )
    pp.process_many(rows, raw, out, progress=False)

    feats, clips = probe.extract_features(
        PatchMeanEncoder(), out, clips_per_volume=3, size=64, batch_size=2, progress=False
    )
    assert feats.shape == (6, 6) and len(clips) == 6
    assert set(clips.columns) >= {"id", "split", "modality", "start", "depth", "slice_height"}
    assert clips["slice_height"].between(0, 1).all()
    heights = clips[clips["id"] == clips["id"][0]]["slice_height"].tolist()
    assert heights == sorted(heights) and heights[0] < heights[-1]  # bottom -> top
