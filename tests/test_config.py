# tests/test_config.py
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
