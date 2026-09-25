import matplotlib.image as mpimg
import numpy as np
import torch

from mri_jepa.viz import plot_clip_grid


def _clip(t=16, size=64):
    yy, xx = np.mgrid[:size, :size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size / 3) ** 2
    mask = np.repeat(disk[None], t, axis=0)
    grey = np.where(mask, 0.3, -1.0).astype(np.float32)
    video = torch.from_numpy(grey)[:, None].repeat(1, 3, 1, 1)
    return video, torch.from_numpy(mask)


def test_writes_a_png(tmp_path):
    video, mask = _clip()
    out = plot_clip_grid(video, mask, tmp_path / "sub" / "grid.png", title="test", first_slice=7)
    img = mpimg.imread(out)
    assert out.is_file() and img.ndim == 3 and img.shape[0] > 500 and img.shape[1] > 500


def test_short_clip_and_no_mask(tmp_path):
    video, _ = _clip(t=5)
    out = plot_clip_grid(video[:, 0].numpy(), None, tmp_path / "grid.png")  # (T, H, W), 5 < 16
    assert out.is_file()
