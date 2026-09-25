import pytest

from mri_jepa import paths


def test_env_var_sets_the_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "data"))
    assert paths.data_root() == tmp_path / "data"
    assert paths.raw_dir() == tmp_path / "data" / "raw"
    assert paths.processed_dir() == tmp_path / "data" / "processed"


def test_missing_setting_gives_a_helpful_error(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="MRI_JEPA_DATA is not set"):
        paths.data_root()
    with pytest.raises(RuntimeError):
        paths.ensure_data_root()


def test_ensure_data_root_creates_folder(tmp_path):
    root = paths.ensure_data_root(tmp_path / "data")
    assert root.is_dir()


def test_ensure_data_root_refuses_when_drive_missing(tmp_path):
    # Simulates the external drive being unplugged: the parent folder does not exist.
    with pytest.raises(FileNotFoundError, match="external drive"):
        paths.ensure_data_root(tmp_path / "not-mounted" / "data")
