# tests/test_manifest.py
from structurezyme.manifest import Manifest, StepRecord

def test_load_missing_returns_empty(tmp_path):
    m = Manifest.load(tmp_path / "manifest.json")
    assert m.steps == {}

def test_set_save_load_roundtrip(tmp_path):
    p = tmp_path / "manifest.json"
    m = Manifest.load(p)
    m.set("chai", StepRecord(status="OK", input_hash="abc", config_hash="def"))
    m.save(p)
    m2 = Manifest.load(p)
    rec = m2.get("chai")
    assert rec.status == "OK"
    assert rec.input_hash == "abc"
    assert rec.config_hash == "def"
