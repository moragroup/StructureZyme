"""File-based locking to guard concurrent writes to shared resources.

Multiple users/jobs on a shared host write into the same Boltz cache and
Squidly weights directories. ``shared_lock`` wraps a ``filelock.FileLock`` on
``<path>.lock`` so those writes are serialized across processes.
"""
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock


@contextmanager
def shared_lock(path, timeout: float = -1):
    """Acquire an inter-process lock guarding ``path``.

    Args:
        path: The resource to guard; the lock file is ``<path>.lock``.
        timeout: Seconds to wait for the lock; ``-1`` waits indefinitely.

    Yields:
        The acquired ``FileLock`` (``lock.is_locked`` is True inside the block).
    """
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path), timeout=timeout)
    with lock:
        yield lock
