"""Unit tests for FastRelax's subprocess seam (no pyrosetta required).

These mock subprocess.run / the filesystem so they run in the plain
structurezyme env with no RosettaFastRelax venv present.
"""
import json
import subprocess
from pathlib import Path

import pytest

from structurezyme.steps import fastrelax_step
from structurezyme.steps.fastrelax_step import (
    FastRelax,
    _parse_worker_stdout,
    _resolve_fastrelax_python,
    _worker_path,
)


def test_worker_path_points_at_worker_file():
    p = _worker_path()
    assert p.name == "_fastrelax_worker.py"
    assert p.exists()


def test_resolve_python_honors_env_override(tmp_path, monkeypatch):
    env_dir = tmp_path / "myenv"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    monkeypatch.setenv("FASTRELAX_ENV", str(env_dir))
    assert _resolve_fastrelax_python() == env_dir / "bin" / "python"


def test_resolve_python_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTRELAX_ENV", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="pyrosetta env python not found"):
        _resolve_fastrelax_python()


def test_parse_stdout_happy_path():
    out = 'noise line\n{"relaxed_path": "/x/a_relaxed.pdb", "score": -12.5}\n'
    path, score = _parse_worker_stdout(out)
    assert path == "/x/a_relaxed.pdb"
    assert score == -12.5


def test_parse_stdout_error_key_raises():
    with pytest.raises(RuntimeError, match="worker error: boom"):
        _parse_worker_stdout('{"error": "boom"}')


def test_parse_stdout_non_json_raises():
    with pytest.raises(RuntimeError, match="not JSON"):
        _parse_worker_stdout("not json at all")


def test_parse_stdout_empty_raises():
    with pytest.raises(RuntimeError, match="no output"):
        _parse_worker_stdout("   \n  \n")


def test_relax_one_invokes_worker_and_parses(tmp_path, monkeypatch):
    """_relax_one builds a spec, calls the worker, and returns its result --
    with subprocess.run and _prepare_pdb_for_pose mocked (no pyrosetta)."""
    fr = FastRelax(output_dir=str(tmp_path))

    # Pretend the venv python exists.
    env_python = tmp_path / "env" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.touch()
    monkeypatch.setenv("FASTRELAX_ENV", str(tmp_path / "env"))

    # Stub the in-process RDKit prep to avoid needing real ligands/params.
    prepared = tmp_path / "a.for_pose.pdb"
    prepared.touch()
    monkeypatch.setattr(
        FastRelax, "_prepare_pdb_for_pose",
        lambda self, p, s, c: (prepared, "X01", None),
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        spec = json.loads(Path(cmd[-1]).read_text())
        captured["spec"] = spec
        out = json.dumps(
            {"relaxed_path": spec["out_pdb"], "score": -9.0}
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    relaxed, score = fr._relax_one("/data/a_0_chai.pdb", "CCO", None)

    assert score == -9.0
    assert relaxed.endswith("a_0_chai_relaxed.pdb")
    assert captured["cmd"][0] == str(env_python)
    assert captured["cmd"][1].endswith("_fastrelax_worker.py")
    assert captured["spec"]["ligand_resname"] == "X01"
    assert captured["spec"]["mode"] == "ligand_focused"


def test_relax_one_worker_failure_raises(tmp_path, monkeypatch):
    """Non-zero worker exit -> RuntimeError (execute() then swallows it)."""
    fr = FastRelax(output_dir=str(tmp_path))
    env_python = tmp_path / "env" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.touch()
    monkeypatch.setenv("FASTRELAX_ENV", str(tmp_path / "env"))
    prepared = tmp_path / "a.for_pose.pdb"
    prepared.touch()
    monkeypatch.setattr(
        FastRelax, "_prepare_pdb_for_pose",
        lambda self, p, s, c: (prepared, "X01", None),
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 1, stdout="", stderr="rosetta boom"
        ),
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        fr._relax_one("/data/a_0_chai.pdb", "CCO", None)
