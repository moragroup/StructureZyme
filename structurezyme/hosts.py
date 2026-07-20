# structurezyme/hosts.py
import os
from importlib import resources
import yaml
from .config import RunConfig

def _all_profiles() -> dict:
    with resources.files("structurezyme").joinpath("hosts.yml").open() as fh:
        return yaml.safe_load(fh)

def load_host_profile(name: str | None = None) -> dict:
    name = name or os.environ.get("STRUCTUREZYME_HOST", "default")
    profiles = _all_profiles()
    if name not in profiles:
        raise KeyError(f"Unknown host profile {name!r}; have {list(profiles)}")
    return profiles[name]

_SENTINEL_PLACER = "/mnt/labs/data/mora/software/PLACER/env"

def apply_host_defaults(cfg: RunConfig, name: str | None = None) -> RunConfig:
    prof = load_host_profile(name)
    p = cfg.paths
    if not p.output_root:
        p.output_root = prof.get("output_root", p.output_root)
    if not p.boltz_cache_dir:
        p.boltz_cache_dir = prof.get("boltz_cache_dir", p.boltz_cache_dir)
    if p.squidly_weights_dir is None:
        p.squidly_weights_dir = prof.get("squidly_weights_dir")
    if p.placer_env_path == _SENTINEL_PLACER and "placer_env_path" in prof:
        p.placer_env_path = prof["placer_env_path"]
    return cfg
