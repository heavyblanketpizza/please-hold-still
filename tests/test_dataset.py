"""The slice-clip Dataset, on phantoms run through the real preprocessing."""

import numpy as np
import pytest
import torch
from conftest import make_phantom
from torch.utils.data import DataLoader

from mri_jepa.data import preprocess as pp
from mri_jepa.data.dataset import MRISliceClipDataset, center_pad_crop, find_volumes


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    """Three preprocessed phantom volumes (one per modality)."""
    tmp = tmp_path_factory.mktemp("ds")
    raw, out = tmp / "raw", tmp / "processed"
    rows = []
    for i, modality in enumerate(("T1w", "T2w", "FLAIR")):
        vid = f"ds00000{i}__sub-01_{modality}"
        f = make_phantom(raw, vid, seed=i)
        rows.append(
            {
                "id": vid,
                "dataset_id": f"ds00000{i}",
                "modality": modality,
                "image": f["image"].name,
                "anat_mask": f["anat"].name,
                "anon_mask": f["anon"].name,
            }
        )
    results = pp.process_many(rows, raw, out, progress=False)
    assert [s for _, s, _ in results] == ["ok"] * 3
    (out / "ds000000" / "._junk.json").write_text("not json")  # macOS litter
    return out


def test_find_volumes_skips_junk_and_filters(processed):
    assert len(find_volumes(processed)) == 3
    assert [v["modality"] for v in find_volumes(processed, ["T2w"])] == ["T2w"]


def test_item_shapes_and_types(processed):
    item = MRISliceClipDataset(processed)[0]
    assert item["video"].shape == (16, 3, 256, 256) and item["video"].dtype == torch.float32
    assert item["mask"].shape == (16, 256, 256) and item["mask"].dtype == torch.bool
    assert isinstance(item["id"], str) and item["modality"] in {"T1w", "T2w", "FLAIR"}


def test_three_channels_are_identical_grey(processed):
    video = MRISliceClipDataset(processed)[1]["video"]
    assert torch.equal(video[:, 0], video[:, 1]) and torch.equal(video[:, 0], video[:, 2])


def test_values_and_background(processed):
    item = MRISliceClipDataset(processed)[0]
    grey, mask = item["video"][:, 0], item["mask"]
    assert grey.min() >= -1 and grey.max() <= 1
    assert mask.any()
    assert (grey[~mask] == -1).all()  # outside anatomy (incl. padding) is background


def test_clip_matches_the_stored_slices(processed):
    ds = MRISliceClipDataset(processed, size=256)
    item = ds.clip(0, start=5)
    info = ds.items[0]
    stored = np.load(f"{info['folder']}/{info['files']['image']}")[5:21].astype(np.float32)
    expected = center_pad_crop(stored, 256, -1.0)
    assert np.array_equal(item["video"][:, 0].numpy(), expected)


def test_eval_is_central_and_repeatable(processed):
    ds = MRISliceClipDataset(processed, train=False)
    depth = ds.items[0]["shape"][0]
    starts = {ds[0]["start"] for _ in range(5)}
    assert starts == {(depth - 16) // 2}


def test_train_start_is_random_and_in_range(processed):
    ds = MRISliceClipDataset(processed, train=True)
    depth = ds.items[0]["shape"][0]
    torch.manual_seed(0)
    starts = [ds[0]["start"] for _ in range(50)]
    assert len(set(starts)) > 5
    assert min(starts) >= 0 and max(starts) <= depth - 16


def test_short_volume_is_padded_with_background(processed):
    ds = MRISliceClipDataset(processed, num_frames=64, train=False)
    depth = ds.items[0]["shape"][0]  # ~49 slices < 64
    item = ds[0]
    assert item["video"].shape[0] == 64
    assert (item["video"][depth:] == -1).all() and not item["mask"][depth:].any()


def test_small_size_centre_crops(processed):
    item = MRISliceClipDataset(processed, size=32)[0]
    assert item["video"].shape == (16, 3, 32, 32) and item["mask"].shape == (16, 32, 32)


def test_center_pad_crop():
    a = np.arange(2 * 3 * 6).reshape(2, 3, 6)
    out = center_pad_crop(a, 4, fill=-9)
    assert out.shape == (2, 4, 4)
    # Height 3 -> 4: one row of padding (an odd amount goes at the end).
    # Width 6 -> 4: one column cropped from each side.
    assert (out[:, 3] == -9).all() and (out[:, 0:3] == a[:, :, 1:5]).all()
    out = center_pad_crop(np.ones((1, 2, 2)), 6, fill=0)
    assert out[0, 2:4, 2:4].sum() == 4 and out.sum() == 4  # padded evenly on all sides


def test_dataloader_batches(processed):
    loader = DataLoader(MRISliceClipDataset(processed), batch_size=2)
    batch = next(iter(loader))
    assert batch["video"].shape == (2, 16, 3, 256, 256)
    assert batch["mask"].shape == (2, 16, 256, 256)
    assert len(batch["id"]) == 2


def test_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MRISliceClipDataset(tmp_path)
