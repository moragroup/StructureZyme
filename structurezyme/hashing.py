# structurezyme/hashing.py
import hashlib
import json
from pathlib import Path

def hash_obj(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def hash_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def hash_files(paths) -> str:
    digests = sorted(hash_file(p) for p in paths if Path(p).is_file())
    return hash_obj(digests)
