# tests/test_runner.py
import pandas as pd
import pytest
from structurezyme.config import RunConfig
from structurezyme.runner import Runner
from structurezyme import registry
from structurezyme.registry import ordered_steps

def _cfg(tmp_path):
    csv = tmp_path / "input.csv"
    pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"],
                  "vina_residues": ["1|2"], "substrate_moiety": ["[C]"]}).to_csv(csv, index=False)
    return RunConfig(
        paths={"output_root": str(tmp_path), "boltz_cache_dir": str(tmp_path / "cache"),
               "input_csv": str(csv)},
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

def test_stop_after_halts_and_leaves_downstream_untouched(tmp_path, monkeypatch):
    calls = []
    def fake_runner(ctx, spec):
        calls.append(spec.name)
        return pd.DataFrame({"Entry": [spec.name]})
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner", fake_runner, raising=False)
    cfg = _cfg(tmp_path)

    Runner(cfg).run(stop_after="prepare_files")

    order = ordered_steps()
    cutoff = order.index("prepare_files")
    downstream = order[cutoff + 1:]
    # No step after prepare_files should have executed...
    assert not any(s in calls for s in downstream), f"downstream ran: {calls}"
    # ...and prepare_files itself must have run.
    assert "prepare_files" in calls
    # No manifest record should exist for downstream steps.
    m = Runner(cfg).manifest
    assert m.get("superimpose") is None
    assert m.get("prepare_files").status == "OK"

def test_stop_after_unknown_step_raises(tmp_path, monkeypatch):
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner",
                            lambda ctx, spec: pd.DataFrame({"Entry": [spec.name]}),
                            raising=False)
    cfg = _cfg(tmp_path)
    with pytest.raises(KeyError):
        Runner(cfg).run(stop_after="not_a_step")


# --------------------------------------------------------------------------
# Optional steps (vina / fastrelax / placer) wiring through the DAG.
# These are disabled by default; verify they are fully integrated: when
# enabled they run in dependency order and deliver their frames to the
# consumers that declare them as optional inputs, and when disabled the
# pipeline degrades gracefully (they are skipped, downstream still runs).
# --------------------------------------------------------------------------
def _record_runner():
    """Fake runner recording call order + which input frames each step saw."""
    calls = []
    seen_inputs = {}
    def fake_runner(ctx, spec):
        calls.append(spec.name)
        # Names of upstream steps whose checkpoint frames were available.
        frames = ctx.input_frames(spec)
        seen_inputs[spec.name] = [f["Entry"].iloc[0] for f in frames]
        return pd.DataFrame({"Entry": [spec.name]})
    return calls, seen_inputs, fake_runner

def test_all_optional_steps_enabled_run_in_order_with_inputs(tmp_path, monkeypatch):
    calls, seen, fake = _record_runner()
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner", fake, raising=False)
    cfg = _cfg(tmp_path)
    cfg.steps.vina.enabled = True
    cfg.steps.fastrelax.enabled = True
    cfg.steps.placer.enabled = True

    Runner(cfg).run()

    # All 15 steps ran, including the three optional ones.
    for name in ("vina", "fastrelax", "placer"):
        assert name in calls, f"{name} did not run when enabled"
    # Dependency ordering honored.
    assert calls.index("vina") < calls.index("docking_metrics")
    assert calls.index("fastrelax") < calls.index("superimpose")
    assert calls.index("plip") < calls.index("placer")
    # Consumers received their optional inputs:
    # docking_metrics declares inputs [boltz, vina] -> both present.
    assert "vina" in seen["docking_metrics"] and "boltz" in seen["docking_metrics"]
    # superimpose declares inputs [prepare_files, fastrelax] -> both present.
    assert "fastrelax" in seen["superimpose"] and "prepare_files" in seen["superimpose"]

def test_all_optional_steps_disabled_degrade_gracefully(tmp_path, monkeypatch):
    calls, seen, fake = _record_runner()
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner", fake, raising=False)
    cfg = _cfg(tmp_path)  # defaults: vina/fastrelax/placer disabled

    Runner(cfg).run()

    m = Runner(cfg).manifest
    for name in ("vina", "fastrelax", "placer"):
        assert name not in calls, f"{name} ran while disabled"
        assert m.get(name).status == "SKIPPED_DISABLED"
    # Downstream consumers still ran, using only their present inputs.
    assert "docking_metrics" in calls
    assert seen["docking_metrics"] == ["boltz"], "vina absent -> only boltz frame"
    assert "superimpose" in calls
    assert seen["superimpose"] == ["prepare_files"], "fastrelax absent -> only prepare_files"


def _cfg_multi(tmp_path, mode):
    csv = tmp_path / "input.csv"
    pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["CCO.O"],
                  "Entry": ["P1"], "vina_residues": ["1|2"],
                  "substrate_moiety": ["CO|O"],
                  "substrate_name": ["ethanol|water"]}).to_csv(csv, index=False)
    return RunConfig(
        paths={"output_root": str(tmp_path),
               "boltz_cache_dir": str(tmp_path / "cache"),
               "input_csv": str(csv)},
        runtime={"user": "tester", "run_id": "r1"},
        multi_substrate_mode=mode,
    )


def test_seed_separate_mode_expands_pickle(tmp_path, monkeypatch):
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner",
                            lambda ctx, spec: pd.DataFrame({"Entry": [spec.name]}),
                            raising=False)
    r = Runner(_cfg_multi(tmp_path, "separate"))
    r._seed_and_validate()
    seed = pd.read_pickle(r.layout.checkpoint_path("_input"))
    assert list(seed["Entry"]) == ["P1__s0", "P1__s1"]
    assert list(seed["enzyme_id"]) == ["P1", "P1"]
    assert list(seed["substrate_smiles"]) == ["CCO", "O"]


def test_seed_off_mode_adds_enzyme_id_only(tmp_path, monkeypatch):
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner",
                            lambda ctx, spec: pd.DataFrame({"Entry": [spec.name]}),
                            raising=False)
    r = Runner(_cfg_multi(tmp_path, "off"))
    r._seed_and_validate()
    seed = pd.read_pickle(r.layout.checkpoint_path("_input"))
    assert list(seed["Entry"]) == ["P1"]
    assert list(seed["enzyme_id"]) == ["P1"]
    assert list(seed["substrate_smiles"]) == ["CCO.O"]
