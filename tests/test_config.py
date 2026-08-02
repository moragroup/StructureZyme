# tests/test_config.py
import pytest
from pydantic import ValidationError

from structurezyme.config import RunConfig, load_config

def _minimal():
    return RunConfig(paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"})

def test_defaults_enable_all_steps():
    cfg = _minimal()
    assert cfg.is_enabled("chai") is True
    assert cfg.runtime.num_threads == 1
    assert cfg.runtime.run_id  # non-empty default

def test_disable_step():
    cfg = RunConfig(paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"},
                    steps={"vina": {"enabled": False}})
    assert cfg.is_enabled("vina") is False

def test_step_options_roundtrip(tmp_path):
    cfg = RunConfig(paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"},
                    steps={"boltz": {"enabled": True, "use_msa_server": False}})
    p = tmp_path / "run.yml"
    cfg.write(p)
    loaded = load_config(p)
    assert loaded.step_options("boltz")["use_msa_server"] is False

def test_multi_substrate_mode_defaults_off():
    cfg = _minimal()
    assert cfg.multi_substrate_mode == "off"

def test_multi_substrate_mode_accepts_valid_values():
    for mode in ("off", "separate", "together"):
        cfg = RunConfig(
            paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"},
            multi_substrate_mode=mode,
        )
        assert cfg.multi_substrate_mode == mode

def test_multi_substrate_mode_rejects_bad_value():
    with pytest.raises(ValidationError):
        RunConfig(
            paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"},
            multi_substrate_mode="both",
        )

def test_multi_substrate_mode_roundtrips(tmp_path):
    cfg = RunConfig(
        paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"},
        multi_substrate_mode="together",
    )
    p = tmp_path / "run.yml"
    cfg.write(p)
    assert load_config(p).multi_substrate_mode == "together"
