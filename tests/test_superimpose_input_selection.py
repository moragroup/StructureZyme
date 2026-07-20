# tests/test_superimpose_input_selection.py
"""Guard: superimpose must consume the fastrelax-relaxed frame when fastrelax
is enabled, matching the legacy Superimposition.run() behavior; otherwise it
falls back to the prepare_files frame."""
import logging

import pandas as pd

from structurezyme.config import RunConfig
from structurezyme import registry
from structurezyme.runner import RunContext
from structurezyme.paths import run_dir, RunLayout
from structurezyme.manifest import Manifest
from structurezyme.step_runners import _superimpose_input


def _ctx(tmp_path, fastrelax_enabled: bool) -> RunContext:
    cfg = RunConfig(
        paths={"output_root": str(tmp_path), "boltz_cache_dir": str(tmp_path / "c")},
        runtime={"user": "t", "run_id": "r1"},
        steps={"fastrelax": {"enabled": fastrelax_enabled}},
    )
    layout = RunLayout(run_dir(cfg.paths.output_root, cfg.runtime.user, cfg.runtime.run_id))
    layout.create()
    return RunContext(cfg, layout, Manifest(), logging.getLogger("test.superimpose"))


def _write_ckpt(ctx, name, tag):
    df = pd.DataFrame({"src": [tag]})
    df.to_pickle(ctx.checkpoint_path(name))


def test_uses_fastrelax_frame_when_enabled(tmp_path):
    ctx = _ctx(tmp_path, fastrelax_enabled=True)
    spec = registry.STEPS["superimpose"]
    _write_ckpt(ctx, "prepare_files", "prepared")
    _write_ckpt(ctx, "fastrelax", "relaxed")
    df = _superimpose_input(ctx, spec)
    assert list(df["src"]) == ["relaxed"]


def test_falls_back_to_prepare_files_when_fastrelax_disabled(tmp_path):
    ctx = _ctx(tmp_path, fastrelax_enabled=False)
    spec = registry.STEPS["superimpose"]
    _write_ckpt(ctx, "prepare_files", "prepared")
    # even if a stale fastrelax checkpoint exists, disabled -> ignore it
    _write_ckpt(ctx, "fastrelax", "relaxed")
    df = _superimpose_input(ctx, spec)
    assert list(df["src"]) == ["prepared"]


def test_falls_back_to_prepare_files_when_fastrelax_ckpt_missing(tmp_path):
    ctx = _ctx(tmp_path, fastrelax_enabled=True)
    spec = registry.STEPS["superimpose"]
    _write_ckpt(ctx, "prepare_files", "prepared")
    # fastrelax enabled but no checkpoint produced -> fall back safely
    df = _superimpose_input(ctx, spec)
    assert list(df["src"]) == ["prepared"]


def test_registry_superimpose_declares_fastrelax_input():
    spec = registry.STEPS["superimpose"]
    assert "fastrelax" in spec.inputs
    assert "prepare_files" in spec.inputs
