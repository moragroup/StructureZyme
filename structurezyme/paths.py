# structurezyme/paths.py
from dataclasses import dataclass
from pathlib import Path

def run_dir(output_root, user: str, run_id: str) -> Path:
    return Path(output_root) / user / run_id

@dataclass
class RunLayout:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)

    @property
    def config_path(self) -> Path:
        return self.root / "config.yml"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def log_path(self) -> Path:
        return self.logs_dir / "structurezyme.log"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    def step_dir(self, name: str) -> Path:
        return self.root / name

    def checkpoint_path(self, name: str) -> Path:
        return self.checkpoints_dir / f"{name}.pkl"

    def create(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
