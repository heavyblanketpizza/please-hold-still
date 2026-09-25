# mri-jepa

Self-supervised 3D brain-MRI encoder. We continue pretraining Meta's V-JEPA 2.1
(a video JEPA) on the OpenMind dataset by treating a stack of consecutive axial
slices as a video clip. Downstream: frozen-encoder probes, then an M3D-style
small LLM on top.

The owner is new to ML engineering. Explain changes in plain language, keep
each step small, and commit after each step with the tests passing.

## Layout

```
src/mri_jepa/     importable package (src layout; there is NO src/__init__.py)
  paths.py        data-root helpers (SSD location, caches)
  data/openmind.py  OpenMind CSV parsing, subset selection, HF download
  data/preprocess.py  NIfTI -> RAS, 1 mm, cropped, normalised .npy + JSON
  data/dataset.py   MRISliceClipDataset: (T,3,256,256) clips + (T,256,256) masks
  models/vjepa.py   load_vjepa2_1_encoder(): V-JEPA 2.1 via torch.hub + manual checkpoint
scripts/          command-line entry points (thin wrappers around the package)
tests/            pytest; synthetic data only, never needs the SSD or network
```

## Commands

```bash
uv sync                      # create .venv and install everything from uv.lock
uv run pytest                # run all tests (fast, CPU only, no downloads)
uv run ruff check .          # lint
uv run ruff format .         # auto-format
uv add <pkg>                 # add a dependency (updates pyproject.toml + uv.lock)
uv run hf auth login         # once per machine, before downloading OpenMind
```

Pipeline (run on the Mac, in order):

```bash
uv run python scripts/download_openmind.py --inspect      # CSV only (a few MB) + summary
uv run python scripts/download_openmind.py --n 200 --dry-run   # show sizes, download nothing
uv run python scripts/download_openmind.py --n 200        # ~200 T1w/T2w/FLAIR + masks
uv run python scripts/preprocess.py --limit 5             # time a few volumes first
uv run python scripts/preprocess.py --workers 8           # all volumes in the manifest
uv run python scripts/smoke_test_encoder.py               # 1 batch through pretrained ViT-B
```

Offline checks (no data, no weights; needs a local vjepa2 clone):

```bash
uv run python scripts/smoke_test_encoder.py --random-weights --synthetic --hub-repo /path/to/vjepa2
VJEPA2_REPO=/path/to/vjepa2 uv run pytest tests/test_vjepa.py   # otherwise those tests skip
```

## Conventions

- Python 3.12, managed by uv. Commit `uv.lock`. Never `pip install` into the venv.
- All large files live under the data root (see "Hardware and storage"), never in
  git. Paths come from `mri_jepa.paths`, never hard-coded in scripts.
- Scripts must be safe to re-run: skip work already done, write files atomically
  (write to a temp name, then `os.replace`).
- Guard anything that downloads more than 5 GB or runs longer than a few
  minutes. Print the size or estimate and require an explicit flag to proceed.
- Tests use small synthetic volumes built in the test itself (or in
  `tests/conftest.py`). They must run in seconds, offline, on CPU.

## OpenMind data

- HF dataset `AnonRes/OpenMind`. `openneuro_metadata.csv` sits at the repo root.
  Its `image_path`, `anat_mask_path` and `anon_mask_path` (the deface mask) are
  relative to the `OpenMind/` folder. Column names match the authors' loader in
  MIC-DKFZ/nnssl (`dataset_conversion/Dataset001_OpenMind.py`).
- `image_quality_score`: lower is better. The authors filter at 1.5–3.5.
- The download step writes `<root>/raw/subset_manifest.csv` (id, dataset_id,
  modality, image, anat_mask, anon_mask, image_quality_score). Paths in it are
  relative to `<root>/raw`. Later steps read this manifest, not the big CSV.

## Preprocessed data format (`<root>/processed/<dataset_id>/`)

- `<id>_image.npy`: float16, shape (Z, Y, X), RAS orientation (z = inferior→
  superior, y = posterior→anterior, x = left→right), 1 mm isotropic, cropped to
  the anatomy-mask bounding box. Clipped at the 1st/99th percentile (measured
  inside the anatomy mask) and scaled to [-1, 1]. Outside the mask it is -1.
- `<id>_anat.npy` / `<id>_anon.npy`: uint8 0/1 anatomy / deface mask, same shape.
  The anon file is missing when the source has no deface mask.
- `<id>.json`: sidecar (shape, spacing, modality, source dataset, crop box,
  affine, clip values). It is written last, so it doubles as the "done" marker.
- Axial slice k is `image[k]`. Load with `np.load(path, mmap_mode="r")` to read
  a few slices without pulling the whole volume into memory.
- `MRISliceClipDataset` returns `video` (T, 3, 256, 256) float32 and `mask`
  (T, 256, 256) bool. It centre-pads/crops in-plane at 1 mm and never stretches.
  Frame row = posterior→anterior and column = left→right (no flips), so plot
  with `origin="lower"`. The V-JEPA encoder wants (B, 3, T, H, W): use
  `video.permute(0, 2, 1, 3, 4)` on a batch.

## V-JEPA 2.1

- Always load through `mri_jepa.models.vjepa`. Never call `torch.hub.load(...,
  pretrained=True)` directly: upstream's `VJEPA_BASE_URL` points at
  `http://localhost:8300`, so it fails. The helper builds the architecture with
  `pretrained=False` from pinned commit `VJEPA_COMMIT`, downloads
  `https://dl.fbaipublicfiles.com/vjepa2/<name>.pt` itself, and loads the
  `ema_encoder` weights (plus `predictor` if asked).
- ViT-B = `vjepa2_1_vit_base_384`, 87M params, patch 16, tubelet 2, dim 768.
  Trained at 384 px, but RoPE lets it run at 256. Input is (B, 3, T, H, W),
  output is (B, T/2 · H/16 · W/16, 768). For 16×256×256 that is 2048 tokens.
- vjepa2's `src/` and `app/` are namespace packages. Our repo's `src/` merges
  with theirs harmlessly, because we only have `src/mri_jepa`. Never add
  `src/__init__.py` or top-level packages named `hub`, `models`, `masks`,
  `utils` or `datasets` under `src/`.

## Hardware and storage

- Local compute: MacBook, Apple silicon, 96 GB unified memory. PyTorch runs on
  the `mps` device. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so ops that MPS lacks
  fall back to CPU instead of crashing.
- Data root: external SSD at `/Volumes/Just for Fun/mri-jepa-data`, which you can
  override with `MRI_JEPA_DATA=/some/other/folder`. The path contains spaces,
  so always quote it in the shell. If the SSD is unplugged,
  `paths.ensure_data_root()` raises instead of writing to the internal disk.
- Caches redirected onto the data root: `HF_HUB_CACHE` and `HF_XET_CACHE` →
  `<root>/hf_cache/` (not `HF_HOME`, which would hide the `hf auth login` token),
  `TORCH_HOME` → `<root>/torch_home`.
- macOS writes `._*` "AppleDouble" files on non-APFS drives. Code that lists
  files must ignore names that start with `._`.
- Cloud Claude sessions cannot reach huggingface.co or dl.fbaipublicfiles.com
  and have no SSD. Real downloads and model runs happen on the Mac.
