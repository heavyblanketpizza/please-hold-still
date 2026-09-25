from pathlib import Path

import pytest

from mri_jepa import paths


def test_env_var_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "data"))
    assert paths.data_root() == tmp_path / "data"
    assert paths.raw_dir() == tmp_path / "data" / "raw"
    assert paths.processed_dir() == tmp_path / "data" / "processed"


def test_default_is_on_the_ssd(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    assert paths.data_root() == Path("/Volumes/Just for Fun/mri-jepa-data")


def test_ensure_data_root_creates_folder(tmp_path):
    root = paths.ensure_data_root(tmp_path / "data")
    assert root.is_dir()


def test_ensure_data_root_refuses_when_drive_missing(tmp_path):
    # Simulates the SSD being unplugged: the parent folder does not exist.
    with pytest.raises(FileNotFoundError, match="SSD"):
        paths.ensure_data_root(tmp_path / "not-mounted" / "data")
