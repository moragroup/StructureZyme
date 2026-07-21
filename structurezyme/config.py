# structurezyme/config.py
import getpass
from datetime import datetime
from pathlib import Path
import yaml
from pydantic import BaseModel, ConfigDict, Field

class StepConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True

class StepsConfig(BaseModel):
    squidly: StepConfig = StepConfig()
    chai: StepConfig = StepConfig()
    boltz: StepConfig = StepConfig()
    vina: StepConfig = StepConfig(enabled=False)
    docking_metrics: StepConfig = StepConfig()
    prepare_files: StepConfig = StepConfig()
    fastrelax: StepConfig = StepConfig(enabled=False)
    superimpose: StepConfig = StepConfig()
    protein_rmsd: StepConfig = StepConfig()
    ligand_rmsd: StepConfig = StepConfig()
    geometric_filter: StepConfig = StepConfig()
    fpocket: StepConfig = StepConfig()
    ligand_sasa: StepConfig = StepConfig()
    plip: StepConfig = StepConfig()
    placer: StepConfig = StepConfig(enabled=False)

class PathsConfig(BaseModel):
    output_root: str = ""
    boltz_cache_dir: str = ""
    input_csv: str = ""
    squidly_weights_dir: str | None = None
    placer_env_path: str = "/mnt/labs/data/mora/software/PLACER/env"

class RuntimeConfig(BaseModel):
    user: str = Field(default_factory=getpass.getuser)
    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))
    num_threads: int = 1
    force: list[str] = Field(default_factory=list)

class RunConfig(BaseModel):
    paths: PathsConfig
    runtime: RuntimeConfig = RuntimeConfig()
    steps: StepsConfig = StepsConfig()

    def validate_paths(self) -> "RunConfig":
        """Assert required paths are set. Call AFTER host defaults are applied.

        ``output_root`` and ``boltz_cache_dir`` may be left empty at
        construction so that a host profile (see
        ``structurezyme.hosts.apply_host_defaults``) can fill them. This gate
        must be invoked at the real run boundary (CLI / Runner start) once
        those defaults have been merged.
        """
        missing = [n for n in ("output_root", "boltz_cache_dir", "input_csv")
                   if not getattr(self.paths, n)]
        if missing:
            raise ValueError(
                f"RunConfig.paths missing required path(s): {', '.join(missing)}. "
                "Set them explicitly or apply a host profile via "
                "structurezyme.hosts.apply_host_defaults."
            )
        return self

    def _step(self, name: str) -> StepConfig:
        return getattr(self.steps, name)

    def is_enabled(self, name: str) -> bool:
        return self._step(name).enabled

    def step_options(self, name: str) -> dict:
        return self._step(name).model_dump()

    def write(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.model_dump(), fh, sort_keys=False)

def load_config(path) -> RunConfig:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return RunConfig(**data)
