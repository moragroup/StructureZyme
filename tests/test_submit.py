# tests/test_submit.py
import pytest
from structurezyme.cli import render_sbatch, main


def _cfg(tmp_path):
    cfg = tmp_path / "run.yml"
    cfg.write_text("paths:\n  output_root: /o\n  boltz_cache_dir: /c\n")
    return cfg


def test_render_sbatch_contains_run_command(tmp_path):
    cfg = _cfg(tmp_path)
    script = render_sbatch(str(cfg), host="default", job_name="sz",
                           partition="gpu", gpus=1, time_limit="24:00:00")
    assert "structurezyme run --config" in script
    assert "--partition=gpu" in script
    assert "--gres=gpu:1" in script


def test_render_sbatch_includes_config_host_jobname_time(tmp_path):
    cfg = _cfg(tmp_path)
    script = render_sbatch(str(cfg), host="myhost", job_name="myjob",
                           partition="short", gpus=2, time_limit="01:00:00")
    assert str(cfg) in script
    assert "--host myhost" in script
    assert "--job-name=myjob" in script
    assert "--time=01:00:00" in script
    assert "--gres=gpu:2" in script


def test_submit_dry_run_prints_script(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    rc = main(["submit", "--config", str(cfg), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "structurezyme run --config" in out
    assert "#SBATCH" in out
