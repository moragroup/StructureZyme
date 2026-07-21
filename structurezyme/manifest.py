# structurezyme/manifest.py
import json
import os
from pathlib import Path
from pydantic import BaseModel, Field

class StepRecord(BaseModel):
    status: str = "PENDING"
    pkl_path: str | None = None
    input_hash: str | None = None
    config_hash: str | None = None
    wall_time_s: float | None = None
    mem_mb: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

class Manifest(BaseModel):
    steps: dict[str, StepRecord] = Field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "Manifest":
        path = Path(path)
        if not path.is_file():
            return cls()
        with open(path) as fh:
            return cls(**json.load(fh))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(self.model_dump(), fh, indent=2)
        os.replace(tmp, path)

    def get(self, name: str) -> StepRecord | None:
        return self.steps.get(name)

    def set(self, name: str, record: StepRecord) -> None:
        self.steps[name] = record
