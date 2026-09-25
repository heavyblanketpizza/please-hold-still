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

Large files (downloads, preprocessed volumes, checkpoints) go under one data
folder, ideally on an external drive. Point the code at it with
`export MRI_JEPA_DATA=/path/to/folder` (add that line to `~/.zshrc` so every
terminal has it).

See `CLAUDE.md` for conventions and the full command list.
