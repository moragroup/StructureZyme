import json
import pytest
from pathlib import Path
from structurezyme.cli import main
from structurezyme.config import load_config

def test_init_writes_template(tmp_path):
    out = tmp_path / "run.yml"
    rc = main(["init", "--output", str(out)])
    assert rc == 0
    assert out.is_file()
    text = out.read_text()
    assert "output_root" in text and "steps" in text

def test_status_on_missing_run_dir_returns_nonzero(tmp_path):
    rc = main(["status", "--run-dir", str(tmp_path / "nope")])
    assert rc != 0

def test_init_template_roundtrips_and_validate_paths_raises(tmp_path):
    out = tmp_path / "run.yml"
    assert main(["init", "--output", str(out)]) == 0
    cfg = load_config(out)
    # Template leaves output_root/boltz_cache_dir empty, so the run-boundary
    # gate must reject it until host defaults or explicit values are supplied.
    with pytest.raises(ValueError):
        cfg.validate_paths()

def test_status_prints_rows_and_returns_zero(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "steps": {
            "squidly": {"status": "OK", "wall_time_s": 12.3},
            "chai": {"status": "PENDING"},
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    rc = main(["status", "--run-dir", str(run_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "squidly" in out and "OK" in out and "12.3s" in out
    assert "chai" in out and "PENDING" in out

def test_unknown_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code != 0

def test_run_missing_required_config_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main(["run"])
    assert exc.value.code != 0

class _FakeRunner:
    last = {}
    def __init__(self, cfg):
        _FakeRunner.last["cfg"] = cfg
    def run(self, stop_after=None):
        _FakeRunner.last["stop_after"] = stop_after

def _write_min_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    from structurezyme.config import RunConfig
    RunConfig(paths={"output_root": str(run_dir), "boltz_cache_dir": str(run_dir / "c")}
              ).write(run_dir / "config.yml")
    return run_dir

def test_step_default_stops_after_named_module(tmp_path, monkeypatch):
    import structurezyme.cli as cli
    monkeypatch.setattr(cli, "Runner", _FakeRunner)
    run_dir = _write_min_run_dir(tmp_path)
    rc = main(["step", "prepare_files", "--run-dir", str(run_dir)])
    assert rc == 0
    assert _FakeRunner.last["stop_after"] == "prepare_files"
    assert "prepare_files" in _FakeRunner.last["cfg"].runtime.force

def test_step_continue_runs_full_pipeline(tmp_path, monkeypatch):
    import structurezyme.cli as cli
    monkeypatch.setattr(cli, "Runner", _FakeRunner)
    run_dir = _write_min_run_dir(tmp_path)
    rc = main(["step", "prepare_files", "--run-dir", str(run_dir), "--continue"])
    assert rc == 0
    assert _FakeRunner.last["stop_after"] is None
    assert "prepare_files" in _FakeRunner.last["cfg"].runtime.force
