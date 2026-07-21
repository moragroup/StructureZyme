# tests/test_hosts.py
import pytest
from structurezyme.config import RunConfig
from structurezyme.hosts import load_host_profile, apply_host_defaults

def test_default_profile_has_keys():
    prof = load_host_profile("default")
    assert "boltz_cache_dir" in prof

def test_apply_does_not_override_explicit(tmp_path):
    cfg = RunConfig(paths={"output_root": "/explicit", "boltz_cache_dir": "/explicit/cache"})
    out = apply_host_defaults(cfg, "default")
    assert out.paths.output_root == "/explicit"
    assert out.paths.boltz_cache_dir == "/explicit/cache"

def test_empty_construct_then_host_apply_fills_paths():
    # Multi-user flow: construct with no core paths, let the host profile fill.
    cfg = RunConfig(paths={})
    assert cfg.paths.output_root == ""
    assert cfg.paths.boltz_cache_dir == ""
    out = apply_host_defaults(cfg, "default")
    prof = load_host_profile("default")
    assert out.paths.output_root == prof["output_root"]
    assert out.paths.boltz_cache_dir == prof["boltz_cache_dir"]
    # validate_paths passes once defaults are applied
    out.paths.input_csv = "/some/in.csv"
    out.validate_paths()

def test_validate_paths_raises_when_core_paths_missing():
    cfg = RunConfig(paths={})
    with pytest.raises(ValueError, match="missing required path"):
        cfg.validate_paths()

def test_validate_paths_returns_self_when_ok():
    cfg = RunConfig(paths={"output_root": "/o", "boltz_cache_dir": "/c", "input_csv": "/in.csv"})
    assert cfg.validate_paths() is cfg
