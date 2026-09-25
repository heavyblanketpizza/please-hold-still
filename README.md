# mri-jepa

Continued self-supervised pretraining of Meta's V-JEPA 2.1 video encoder on 3D
brain MRI (the OpenMind dataset), treating consecutive axial slices as video
frames.

## Setup (Mac)

```bash
# once: install uv  →  https://docs.astral.sh/uv/getting-started/installation/
uv sync          # creates .venv/ and installs the exact versions in uv.lock
uv run pytest    # should print only dots and "passed"
```

Data goes to `/Volumes/Just for Fun/mri-jepa-data` by default. To use a different
folder, set `export MRI_JEPA_DATA=/path/to/folder`.

See `CLAUDE.md` for conventions and the full command list.
