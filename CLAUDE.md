# please-hold-still

Self-supervised 3D brain-MRI encoder. We continue pretraining Meta's V-JEPA 2.1
(a video JEPA) on the OpenMind dataset by treating a stack of consecutive axial
slices as a video clip. Downstream: frozen-encoder probes, then an M3D-style
small LLM on top.

The owner is new to ML engineering. Explain changes in plain language, keep
each step small, and commit after each step with the tests passing.

## Layout

```
src/please_hold_still/  importable package (src layout; there is NO src/__init__.py)
  paths.py        data-root helpers ($PLEASE_HOLD_STILL_DATA, caches)
  data/openmind.py  OpenMind CSV parsing, subset selection, HF download
  data/preprocess.py  NIfTI -> RAS, 1 mm, cropped, normalised .npy + JSON
  data/dataset.py   MRISliceClipDataset: (T,3,256,256) clips + (T,256,256) masks
  models/vjepa.py   load_vjepa2_1_encoder(): V-JEPA 2.1 via torch.hub + manual checkpoint
  viz.py            plot_clip_grid(): 4x4 slice grid with mask overlay -> PNG
  notify.py         macOS notification + sound when long scripts finish
  masking.py        ForegroundBlockMasker: V-JEPA 2.1 multi-block masks on anatomy only
  jepa.py           JEPA: student + EMA teacher + predictor; one training step's loss
  probe.py          frozen-encoder features + linear probes + before/after table
  training.py       training loop: warm-up, EMA, checkpoints/resume, disk check
  data/splits.py    split_of(): stable 80/20 train/test split by OpenNeuro study
scripts/          command-line entry points (thin wrappers around the package)
tests/            pytest; synthetic data only, never needs the SSD or network
docs/             images used by README.md (small PNGs only)
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
caffeinate -i uv run python scripts/download_openmind.py --n 200   # ~200 volumes; Mac stays awake
uv run python scripts/download_openmind.py --status       # progress, from another terminal
uv run python scripts/preprocess.py --limit 5             # time a few volumes first
uv run python scripts/preprocess.py --workers 8           # all volumes in the manifest
uv run python scripts/smoke_test_encoder.py               # 1 batch through pretrained ViT-B
uv run python scripts/benchmark_train_step.py --bf16     # s/step + batch size, no data needed
uv run python scripts/visualize_clip.py --random 4        # PNG grids in outputs/ (git-ignored)
uv run python scripts/evaluate_encoder.py --tag baseline  # probes on the original encoder
uv run python scripts/train_jepa.py --run-name overfit --steps 200 --max-volumes 8 --predictor-only-steps 50
caffeinate -i uv run python scripts/train_jepa.py --run-name run1 --steps 1000
uv run python scripts/evaluate_encoder.py --tag run1 --checkpoint "$PLEASE_HOLD_STILL_DATA/runs/run1/encoder_last.pt"
uv run python scripts/compare_models.py baseline run1     # before/after table + chart
```

Offline checks (no data, no weights; needs a local vjepa2 clone):

```bash
uv run python scripts/smoke_test_encoder.py --random-weights --synthetic --hub-repo /path/to/vjepa2
VJEPA2_REPO=/path/to/vjepa2 uv run pytest tests/test_vjepa.py   # otherwise those tests skip
```

## Conventions

- Python 3.12, managed by uv. Commit `uv.lock`. Never `pip install` into the venv.
- All large files live under the data root (see "Hardware and storage"), never in
  git. Paths come from `please_hold_still.paths`, never hard-coded in scripts.
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
  MIC-DKFZ/nnssl (`dataset_conversion/Dataset001_OpenMind.py`), and the real
  file (checked 2026-09-26). Scans from one session can share a mask file, so
  200 volumes need fewer than 600 files (563 in our subset).
- **The raw anatomy mask (`fb_mask`) is stored the other way round: 1 =
  background, 0 = head.** nnssl's loader takes the foreground as `1 - anat`,
  and on our 200 real scans the head was ~40× brighter outside the stored mask.
  `preprocess.load_anatomy_mask()` flips it on load, and `preprocess_volume`
  raises "wrong way round" if the mask ever covers the darker part of an image.
  The raw deface mask is the usual way: 1 = defaced (nnssl uses `1 - anon`).
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

- Always load through `please_hold_still.models.vjepa`. Never call
  `torch.hub.load(..., pretrained=True)` directly: upstream's `VJEPA_BASE_URL`
  points at `http://localhost:8300`, so it fails. The helper builds the architecture with
  `pretrained=False` from pinned commit `VJEPA_COMMIT`, downloads
  `https://dl.fbaipublicfiles.com/vjepa2/<name>.pt` itself, and loads the
  `ema_encoder` weights (plus `predictor` if asked).
- ViT-B = `vjepa2_1_vit_base_384`, 87M params, patch 16, tubelet 2, dim 768.
  Trained at 384 px, but RoPE lets it run at 256. Input is (B, 3, T, H, W),
  output is (B, T/2 · H/16 · W/16, 768). For 16×256×256 that is 2048 tokens.
- vjepa2's `src/` and `app/` are namespace packages. Our repo's `src/` merges
  with theirs harmlessly, because we only have `src/please_hold_still`. Never add
  `src/__init__.py` or top-level packages named `hub`, `models`, `masks`,
  `utils` or `datasets` under `src/`.
- Predictor grid pitfall: the predictor turns flat token indices into
  (t, row, col) using the grid it was built for (24×24 at 384 px). Unlike the
  encoder, it is not told the real grid, so at 256 px every position is silently
  wrong. Call `set_predictor_grid(predictor, H // 16, T)` before using it.
  `JEPA.forward` does this for you on every call.

## Hardware and storage

- Local compute: MacBook, Apple silicon, 96 GB unified memory. PyTorch runs on
  the `mps` device. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so ops that MPS lacks
  fall back to CPU instead of crashing.
- Data root: a folder on an external drive, set by `PLEASE_HOLD_STILL_DATA`
  (the owner exports it in `~/.zshrc`), or `--data-root` on any script. Never
  write a machine-specific path into the repo: code, docs and tests take it
  from the environment. There is no default. If the variable is unset,
  `paths.data_root()` raises. If the drive is unplugged,
  `paths.ensure_data_root()` raises instead of writing to the internal disk.
  Drive names can contain spaces, so always quote the path in the shell.
- Caches redirected onto the data root: `HF_HUB_CACHE` and `HF_XET_CACHE` →
  `<root>/hf_cache/` (not `HF_HOME`, which would hide the `hf auth login` token),
  `TORCH_HOME` → `<root>/torch_home`.
- macOS writes `._*` "AppleDouble" files on non-APFS drives. Code that lists
  files must ignore names that start with `._`.
- Cloud Claude sessions cannot reach huggingface.co or dl.fbaipublicfiles.com
  and have no SSD. Real downloads and model runs happen on the Mac.

## Disk usage (per 200-volume experiment, on the data drive)

- raw download: roughly 2–5 GB (`--dry-run` prints the exact number)
- preprocessed: about 10–20 MB per volume, so 2–4 GB
- V-JEPA checkpoint: downloaded once; its size is printed first, max 5 GB
- each training run: `checkpoint_last.pt` about 1.7 GB (briefly twice that
  while it is replaced) plus about 0.35 GB per `encoder_stepN.pt`. So about
  5 GB for 1000 steps with `--save-every 250`. The script checks free space
  before starting.
- eval features: a few MB per evaluated model
- Training tests (`VJEPA2_REPO` set) write GBs to pytest's temp folder. pytest
  keeps them only for failed tests (`tmp_path_retention_policy`).

## Status and next steps (keep this section current)

Built and tested on synthetic data: download (with `--status` and a
notification when done), preprocessing, Dataset (with a stable train/test
split by study), V-JEPA loader and smoke test,
visualisation, foreground masking, the JEPA training step (`jepa.py`), the
training script (`train_jepa.py`), and the probe evaluation + comparison
(`evaluate_encoder.py`, `compare_models.py`). The full loop (evaluate → train →
evaluate → compare) has been run end to end on fake volumes, and on the
200 real volumes (run1, below).

Next, on the Mac, in order:

1. **Real data.** Done on the Mac (2026-09-26): `--inspect`, `--n 200
   --dry-run` and the download itself. The real CSV has the expected columns,
   and 200 volumes (67 T1w, 67 T2w, 66 FLAIR from 122 studies, 2.39 GB) are on
   the data drive with a complete `subset_manifest.csv`. The Hugging Face
   download worked without logging in. `preprocess.py --workers 8` then ran
   on all 200 in 30 s (PREPROCESS_VERSION 2, after the anatomy-mask flip
   fix), and `visualize_clip.py` PNGs look right. A few scans cover only a
   slab of the head (smallest crop 55 mm front to back). Still to do:
   `smoke_test_encoder.py`.
2. **Baseline:** done (2026-09-26), `<root>/eval/baseline/`. 200 volumes,
   33 test volumes from 20 studies. Scan type 0.91 balanced accuracy (95%
   range 0.80–1.00, chance 0.33); sex 0.79 (0.64–0.92, chance 0.5); age MAE
   9.8 years (6.9–12.9; guessing the mean gives 11.4, R² 0.24); slice height
   MAE 0.062 of head height (0.055–0.070, chance 0.26, R² 0.93). Age is the
   weakest and has the most room to improve. `smoke_test_encoder.py` passed
   on real data (0.74 s per 2-clip batch on mps).
3. **Sanity run:** done (2026-09-26), `<root>/runs/overfit/` (model files
   since deleted; `log.csv` and `config.json` kept). Loss fell
   0.76 → 0.52 over 200 steps (about 2.5–2.9 s per step at batch 8, fp32).
   Probes on its `encoder_last.pt` (`<root>/eval/overfit/`) match the
   baseline within noise (age MAE 9.4 vs 9.8, others unchanged). That is
   expected: the saved encoder is the EMA teacher, which moves only ~11% of
   the way toward the student in 150 training steps (0.99925^150 ≈ 0.89).
   Clip features changed ~4% (cosine 0.999), so the checkpoint does load.
   In run1 (1000 steps, the first 200 predictor-only) the teacher moves ~45%
   (0.99925^800 ≈ 0.55).
4. **Real run:** done (2026-09-26), `<root>/runs/run1/` (all 200 volumes,
   batch 8, fp32, 46 min, 1.7–2.8 s per step). Loss fell 0.76 → ~0.45 and
   was flat from about step 650. Probes (`<root>/eval/run1/`) vs baseline:
   scan type 0.97 (was 0.91), sex 0.79 (same), age MAE 8.5 years (was 9.8,
   R² 0.34 vs 0.24), slice height 0.059 (was 0.062). All four point the
   right way or stay level, but every 95% range overlaps the baseline's, so
   none is proven yet. Chart: `<root>/eval/compare__baseline__overfit__run1.png`.
   The four `encoder_stepN.pt` snapshots are still on disk.
5. With ~40 test volumes the 95% ranges are wide. A convincing comparison
   probably needs more data (e.g. `--n 2000`). Check the size with the owner first.
6. Later: Meta's 4-layer "deep supervision" targets; bf16 for speed; the
   whole-volume representation for the LLM stage (longer clips, slice stride,
   or pooling several clips).

Design notes for the training step (`jepa.py`): targets are the teacher's last
layer only (768-d, layer-normed). The predictor keeps its pretrained body; only
its two ViT-G-sized output layers are replaced. Loss = L1(hidden) + 0.5 ·
distance-weighted L1(visible). EMA 0.99925. The student is frozen for
`predictor_only_steps` while the new layers catch up.
