# tests/test_locks.py
from structurezyme.locks import shared_lock


def test_shared_lock_acquires_and_releases(tmp_path):
    target = tmp_path / "cache"
    with shared_lock(target) as lock:
        assert lock.is_locked
    assert not lock.is_locked


def test_shared_lock_creates_parent_dir(tmp_path):
    target = tmp_path / "nested" / "sub" / "cache"
    with shared_lock(target) as lock:
        assert lock.is_locked
    assert (tmp_path / "nested" / "sub").is_dir()
