"""PyTorch Dataset: preprocessed MRI volumes as short "videos" of axial slices.

A clip is T consecutive axial slices (default 16) from one volume. Each slice
becomes one video frame of shape (3, 256, 256): the grey values copied into
3 channels, because V-JEPA was trained on RGB video.

Frames are in the stored orientation, so no flips are applied:
row index = posterior→anterior, column index = left→right (patient's right is
on the image's right). One pixel is 1 mm. Slices narrower than `size` are
padded with background (-1); wider ones are centre-cropped. Nothing is
stretched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mri_jepa.data.splits import split_of

BACKGROUND = -1.0


def find_volumes(
    processed_dir: str | Path,
    modalities: list[str] | None = None,
    split: str | None = None,
) -> list[dict]:
    """Read every JSON sidecar under `processed_dir`, sorted by id.

    Each returned dict is the sidecar plus `folder` and `split` ("train" or
    "test", by study; see `splits.py`). `split` keeps only that side. Skips
    macOS `._*` junk files and our own `_*.csv`/`_*.json` bookkeeping files.
    """
    if split not in (None, "train", "test"):
        raise ValueError(f"split must be None, 'train' or 'test', not {split!r}")
    items = []
    for path in sorted(Path(processed_dir).rglob("*.json")):
        if path.name.startswith(("._", "_")):
            continue
        info = json.loads(path.read_text())
        info["split"] = split_of(info["source_dataset"])
        if modalities is not None and info["modality"] not in modalities:
            continue
        if split is not None and info["split"] != split:
            continue
        info["folder"] = str(path.parent)
        items.append(info)
    return sorted(items, key=lambda d: d["id"])


def center_pad_crop(slices: np.ndarray, size: int, fill) -> np.ndarray:
    """(T, H, W) -> (T, size, size): centre-crop axes that are too big, pad ones too small."""
    t, h, w = slices.shape
    out = np.full((t, size, size), fill, dtype=slices.dtype)
    src, dst = [], []
    for n in (h, w):
        if n >= size:  # crop
            a = (n - size) // 2
            src.append(slice(a, a + size))
            dst.append(slice(0, size))
        else:  # pad
            a = (size - n) // 2
            src.append(slice(0, n))
            dst.append(slice(a, a + n))
    out[:, dst[0], dst[1]] = slices[:, src[0], src[1]]
    return out


class MRISliceClipDataset(Dataset):
    """Clips of `num_frames` consecutive axial slices from preprocessed volumes.

    Each item is a dict:
        video     float32 (T, 3, size, size), values in [-1, 1]
        mask      bool    (T, size, size), True where there is anatomy
        id, modality      strings from the sidecar
        start             index of the first slice in the volume
        depth             number of axial slices in the volume

    With `train=True` the clip starts at a random slice (a new one each time
    the item is read). With `train=False` it is the central clip, so
    evaluation is repeatable.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        num_frames: int = 16,
        size: int = 256,
        train: bool = True,
        modalities: list[str] | None = None,
        split: str | None = None,
    ):
        self.items = find_volumes(processed_dir, modalities, split)
        if not self.items:
            where = f" (split={split!r})" if split else ""
            raise FileNotFoundError(f"no preprocessed volumes found under {processed_dir}{where}")
        self.num_frames = num_frames
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def choose_start(self, depth: int) -> int:
        """First slice of the clip, for a volume with `depth` axial slices."""
        last = max(depth - self.num_frames, 0)
        if self.train:
            # torch's RNG is seeded differently in each DataLoader worker, so
            # workers do not all pick the same slices.
            return int(torch.randint(0, last + 1, ()).item())
        return last // 2

    def clip(self, index: int, start: int | None = None) -> dict:
        """The clip for volume `index`. Pass `start` to choose the first slice yourself."""
        info = self.items[index]
        folder = Path(info["folder"])
        # mmap: only the slices we slice out are read from disk.
        image = np.load(folder / info["files"]["image"], mmap_mode="r")
        anat = np.load(folder / info["files"]["anat"], mmap_mode="r")
        depth = image.shape[0]
        if start is None:
            start = self.choose_start(depth)
        stop = min(start + self.num_frames, depth)

        img = np.full((self.num_frames, *image.shape[1:]), BACKGROUND, np.float32)
        msk = np.zeros((self.num_frames, *image.shape[1:]), bool)
        img[: stop - start] = image[start:stop]  # short volumes: extra frames stay background
        msk[: stop - start] = anat[start:stop] > 0

        img = center_pad_crop(img, self.size, BACKGROUND)
        msk = center_pad_crop(msk, self.size, False)
        video = torch.from_numpy(img)[:, None].repeat(1, 3, 1, 1)  # grey -> 3 channels
        return {
            "video": video,
            "mask": torch.from_numpy(msk),
            "id": info["id"],
            "modality": info["modality"],
            "start": start,
            "depth": depth,
        }

    def __getitem__(self, index: int) -> dict:
        return self.clip(index)
