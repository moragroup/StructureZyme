# tests/test_runner.py
import pandas as pd
from structurezyme.config import RunConfig
from structurezyme.runner import Runner
from structurezyme import registry

def _cfg(tmp_path):
    return RunConfig(
        paths={"output_root": str(tmp_path), "boltz_cache_dir": str(tmp_path / "cache")},
        runtime={"user": "tester", "run_id": "r1"},
    )

def test_disabled_step_is_skipped_and_recorded(tmp_path, monkeypatch):
    calls = []
    def fake_runner(ctx, spec):
        calls.append(spec.name)
        return pd.DataFrame({"Entry": [spec.name]})
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner", fake_runner, raising=False)
    cfg = _cfg(tmp_path)
    cfg.steps.vina.enabled = False
    Runner(cfg).run()
    assert "vina" not in calls
    m = Runner(cfg).manifest
    assert m.get("vina").status == "SKIPPED_DISABLED"

def test_checkpoint_hit_skips_second_run(tmp_path, monkeypatch):
    calls = []
    def fake_runner(ctx, spec):
        calls.append(spec.name)
        return pd.DataFrame({"Entry": [spec.name]})
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner", fake_runner, raising=False)
    cfg = _cfg(tmp_path)
    Runner(cfg).run()
    first = len(calls)
    calls.clear()
    Runner(cfg).run()  # second run: everything cached
    assert calls == [], "no step should re-run on a clean cache hit"
    assert first > 0

def test_checkpoint_hit_survives_repeated_runs(tmp_path, monkeypatch):
    calls = []
    def fake_runner(ctx, spec):
        calls.append(spec.name)
        return pd.DataFrame({"Entry": [spec.name]})
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner", fake_runner, raising=False)
    cfg = _cfg(tmp_path)

    Runner(cfg).run()  # first run: everything computed
    assert calls, "first run should compute steps"
    calls.clear()

    Runner(cfg).run()  # second run: everything cached
    assert calls == [], "no step should re-run on 2nd run"
    m2 = Runner(cfg).manifest
    assert m2.get("chai").status == "OK", "enabled step must stay OK after run2"
    calls.clear()

    Runner(cfg).run()  # third run: STILL cached (regression guard)
    assert calls == [], "no step should re-run on 3rd run"
    m3 = Runner(cfg).manifest
    assert m3.get("chai").status == "OK", "enabled step must stay OK after run3"
