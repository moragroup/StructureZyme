# tests/test_integration_resume.py
import pandas as pd
import pytest
from structurezyme.config import RunConfig
from structurezyme.runner import Runner
from structurezyme import registry


def _cfg(tmp_path):
    csv = tmp_path / "input.csv"
    pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"],
                  "vina_residues": ["1|2"], "substrate_moiety": ["[C]"]}).to_csv(csv, index=False)
    return RunConfig(
        paths={"output_root": str(tmp_path), "boltz_cache_dir": str(tmp_path / "c"),
               "input_csv": str(csv)},
        runtime={"user": "t", "run_id": "r"},
    )


def test_crash_then_resume(tmp_path, monkeypatch):
    order = registry.ordered_steps()
    # Crash at an ENABLED step (default config disables vina/fastrelax/placer;
    # a disabled step is skipped and would never raise). Pick the 4th enabled
    # step so there are completed steps before it to prove resume skips them.
    cfg0 = _cfg(tmp_path)
    enabled_order = [s for s in order if cfg0.is_enabled(s)]
    crash_at = enabled_order[3]
    first_enabled = enabled_order[0]
    state = {"crash": True}

    def make(name):
        def _run(ctx, spec):
            if spec.name == crash_at and state["crash"]:
                raise RuntimeError("boom")
            return pd.DataFrame({"Entry": [spec.name]})
        return _run
    for name, spec in registry.STEPS.items():
        monkeypatch.setattr(spec, "runner", make(name), raising=False)

    with pytest.raises(RuntimeError):
        Runner(_cfg(tmp_path)).run()

    r = Runner(_cfg(tmp_path))
    assert r.manifest.get(crash_at).status == "FAILED"
    assert r.manifest.get(first_enabled).status == "OK"

    state["crash"] = False
    ran = []

    def make2(name):
        def _run(ctx, spec):
            ran.append(spec.name)
            return pd.DataFrame({"Entry": [spec.name]})
        return _run
    for name, spec in registry.STEPS.items():
        monkeypatch.setattr(spec, "runner", make2(name), raising=False)
    Runner(_cfg(tmp_path)).run()
    assert first_enabled not in ran      # cached, not re-run
    assert crash_at in ran               # resumed
    assert enabled_order[-1] in ran      # completed to the last enabled step


def test_force_reruns_target(tmp_path, monkeypatch):
    def make(name):
        def _run(ctx, spec):
            return pd.DataFrame({"Entry": [spec.name]})
        return _run
    for name, spec in registry.STEPS.items():
        monkeypatch.setattr(spec, "runner", make(name), raising=False)
    Runner(_cfg(tmp_path)).run()

    ran = []

    def make2(name):
        def _run(ctx, spec):
            ran.append(spec.name)
            return pd.DataFrame({"Entry": [spec.name]})
        return _run
    for name, spec in registry.STEPS.items():
        monkeypatch.setattr(spec, "runner", make2(name), raising=False)
    cfg = _cfg(tmp_path)
    cfg.runtime.force = ["chai"]
    Runner(cfg).run()
    assert "chai" in ran
