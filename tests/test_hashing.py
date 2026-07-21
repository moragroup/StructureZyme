# tests/test_hashing.py
from structurezyme.hashing import hash_obj, hash_file, hash_files

def test_hash_obj_is_key_order_independent():
    assert hash_obj({"a": 1, "b": 2}) == hash_obj({"b": 2, "a": 1})

def test_hash_obj_changes_with_value():
    assert hash_obj({"a": 1}) != hash_obj({"a": 2})

def test_hash_file_and_files(tmp_path):
    p1 = tmp_path / "a.txt"; p1.write_text("hello")
    p2 = tmp_path / "b.txt"; p2.write_text("world")
    assert hash_file(p1) == hash_file(p1)
    assert hash_files([p1, p2]) == hash_files([p2, p1])
