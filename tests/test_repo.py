"""Guard against code that exists on disk but never reaches git."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="not a git checkout"
)
def test_no_source_file_is_ignored_by_git():
    """A too-broad .gitignore rule once hid src/mri_jepa/data/ from every commit."""
    files = [
        str(p.relative_to(ROOT))
        for folder in ("src", "scripts", "tests")
        for p in (ROOT / folder).rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", *files], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    assert ignored == [], f"these code files are ignored by .gitignore: {ignored}"
