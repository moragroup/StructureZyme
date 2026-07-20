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
