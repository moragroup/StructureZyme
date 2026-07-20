# tests/test_hosts.py
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
