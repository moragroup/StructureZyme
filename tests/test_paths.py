# tests/test_paths.py
from pathlib import Path
from structurezyme.paths import run_dir, RunLayout

def test_run_dir_layout(tmp_path):
    rd = run_dir(tmp_path, "alice", "run1")
    assert rd == tmp_path / "alice" / "run1"

def test_runlayout_paths_and_create(tmp_path):
    layout = RunLayout(run_dir(tmp_path, "alice", "run1"))
    layout.create()
    assert layout.logs_dir.is_dir()
    assert layout.checkpoints_dir.is_dir()
    assert layout.manifest_path == layout.root / "manifest.json"
    assert layout.log_path == layout.logs_dir / "structurezyme.log"
    assert layout.checkpoint_path("chai") == layout.checkpoints_dir / "chai.pkl"
    assert layout.step_dir("docking") == layout.root / "docking"
