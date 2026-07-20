# tests/test_seed.py
import pytest
from structurezyme.config import RunConfig, PathsConfig


def _cfg(**paths):
    base = dict(output_root="/tmp/o", boltz_cache_dir="/tmp/b", input_csv="/tmp/in.csv")
    base.update(paths)
    return RunConfig(paths=PathsConfig(**base))


def test_validate_paths_requires_input_csv():
    cfg = _cfg(input_csv="")
    with pytest.raises(ValueError) as e:
        cfg.validate_paths()
    assert "input_csv" in str(e.value)


def test_validate_paths_ok_with_input_csv():
    assert _cfg().validate_paths() is not None
