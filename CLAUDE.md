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

## Hardware and storage

- Local compute: MacBook, Apple silicon, 96 GB unified memory. PyTorch runs on
  the `mps` device. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so ops that MPS lacks
  fall back to CPU instead of crashing.
- Data root: external SSD at `/Volumes/Just for Fun/mri-jepa-data`, which you can
  override with `MRI_JEPA_DATA=/some/other/folder`. The path contains spaces,
  so always quote it in the shell. If the SSD is unplugged,
  `paths.ensure_data_root()` raises instead of writing to the internal disk.
- Caches redirected onto the data root: `HF_HOME` → `<root>/hf_home`,
  `TORCH_HOME` → `<root>/torch_home`.
- macOS writes `._*` "AppleDouble" files on non-APFS drives. Code that lists
  files must ignore names that start with `._`.
- Cloud Claude sessions cannot reach huggingface.co or dl.fbaipublicfiles.com
  and have no SSD. Real downloads and model runs happen on the Mac.
