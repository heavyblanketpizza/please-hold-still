"""Download-step logic, tested on a small fake metadata table (no network)."""

import numpy as np
import pandas as pd
import pytest

from mri_jepa.data import openmind as om


@pytest.fixture
def meta(tmp_path):
    """Fake openneuro_metadata.csv: 10 studies with uneven sizes, 4 modalities."""
    rows = []
    rng = np.random.default_rng(0)
    for d in range(10):
        ds = f"ds{d:06d}"
        for s in range(d + 1):  # study d has d+1 subjects
            for mod in ("T1w", "T2w", "FLAIR", "dwi"):
                rel = f"{ds}/sub-{s:02d}/anat/sub-{s:02d}_{mod}.nii.gz"
                rows.append(
                    {
                        "image_path": rel,
                        "modality": mod,
                        "anat_mask_path": f"anatomy_masks/{rel}",
                        "anon_mask_path": f"deface_masks/{rel}" if s % 2 == 0 else np.nan,
                        "image_quality_score": float(rng.uniform(0, 5)),
                        "unique_id": f"{ds}_{s}_{mod}",
                    }
                )
    # One T1w without an anatomy mask: must never be selected.
    rows.append({"image_path": "ds999999/sub-01/anat/sub-01_T1w.nii.gz", "modality": "T1w"})
    csv = tmp_path / om.METADATA_FILENAME
    pd.DataFrame(rows).to_csv(csv, index=False)
    return om.load_metadata(csv)


def test_load_metadata_adds_dataset_id(meta):
    assert meta.loc[0, "dataset_id"] == "ds000000"


def test_load_metadata_complains_about_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"path": ["x"]}).to_csv(csv, index=False)
    with pytest.raises(KeyError, match="image_path"):
        om.load_metadata(csv)


def test_volume_id_is_readable_and_filename_safe():
    rid = om.volume_id("ds000117/sub-01/ses-mri/anat/sub-01_T1w.nii.gz")
    assert rid == "ds000117__sub-01__ses-mri__anat__sub-01_T1w"
    assert "/" not in rid


def test_even_split_across_modalities(meta):
    subset = om.select_subset(meta, 30)
    assert subset["modality"].value_counts().to_dict() == {"T1w": 10, "T2w": 10, "FLAIR": 10}
    assert "dwi" not in set(subset["modality"])


def test_uneven_total_is_spread(meta):
    counts = om.select_subset(meta, 32)["modality"].value_counts()
    assert counts.sum() == 32 and counts.max() - counts.min() <= 1


def test_round_robin_uses_every_study_before_repeating(meta):
    # 10 studies, so the first 10 T1w picks must come from 10 different studies.
    t1 = om.select_subset(meta, 30).query("modality == 'T1w'")
    assert t1["dataset_id"].nunique() == 10


def test_selection_is_deterministic_and_grows_as_superset(meta):
    small = om.select_subset(meta, 30, seed=1)
    again = om.select_subset(meta, 30, seed=1)
    big = om.select_subset(meta, 60, seed=1)
    assert small["image_path"].tolist() == again["image_path"].tolist()
    assert set(small["image_path"]) <= set(big["image_path"])


def test_rows_without_anatomy_mask_are_skipped(meta):
    subset = om.select_subset(meta, 150)
    assert not subset["image_path"].str.startswith("ds999999").any()


def test_max_iqs_filters(meta):
    subset = om.select_subset(meta, 150, max_iqs=2.0)
    assert (subset["image_quality_score"] <= 2.0).all()


def test_manifest_paths_and_optional_deface_mask(meta):
    manifest = om.build_manifest(om.select_subset(meta, 30), prefix="OpenMind")
    assert list(manifest.columns) == om.MANIFEST_COLUMNS
    assert manifest["image"].str.startswith("OpenMind/ds").all()
    assert manifest["anat_mask"].str.startswith("OpenMind/anatomy_masks/").all()
    has_anon = manifest["anon_mask"] != ""
    assert has_anon.any() and (~has_anon).any()  # fixture has both kinds
    files = om.manifest_files(manifest)
    assert len(files) == 2 * len(manifest) + has_anon.sum()


def test_files_to_download_skips_complete_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "done.nii.gz").write_bytes(b"x" * 10)
    (tmp_path / "a" / "partial.nii.gz").write_bytes(b"x" * 3)
    sizes = {"a/done.nii.gz": 10, "a/partial.nii.gz": 10, "a/new.nii.gz": 7}
    todo = dict(om.files_to_download(sizes, tmp_path))
    assert todo == {"a/partial.nii.gz": 10, "a/new.nii.gz": 7}


def test_download_files_reports_failures(tmp_path):
    def fake_download(path):
        if "bad" in path:
            raise OSError("network down")
        (tmp_path / path).write_bytes(b"ok")

    failed = om.download_files(
        [("good1", 2), ("bad", 2), ("good2", 2)], tmp_path, workers=2, download_fn=fake_download
    )
    assert failed == ["bad"]
    assert (tmp_path / "good1").is_file() and (tmp_path / "good2").is_file()
