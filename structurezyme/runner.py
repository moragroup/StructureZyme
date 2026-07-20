# structurezyme/runner.py
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd

from .config import RunConfig
from .paths import run_dir, RunLayout
from .manifest import Manifest, StepRecord
from .registry import STEPS, StepSpec, ordered_steps
from .hashing import hash_obj, hash_files

REQUIRED_INPUT_COLUMNS: dict[str, list[str]] = {
    "_base": ["Sequence", "substrate_smiles", "Entry"],
    "vina": ["vina_residues"],
    "geometric_filter": ["substrate_moiety"],
}


def _load_input_frame(path: str) -> pd.DataFrame:
    """Load the seed input DataFrame from a CSV or pickle file."""
    if str(path).endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    return pd.read_csv(path)


def missing_input_columns(df: pd.DataFrame, enabled: set[str]) -> dict[str, list[str]]:
    """Return {group: [missing cols]} for the base plus each enabled module."""
    cols = set(df.columns)
    groups = ["_base"] + [m for m in REQUIRED_INPUT_COLUMNS if m != "_base" and m in enabled]
    out: dict[str, list[str]] = {}
    for g in groups:
        miss = [c for c in REQUIRED_INPUT_COLUMNS[g] if c not in cols]
        if miss:
            out[g] = miss
    return out


@dataclass
class RunContext:
    config: RunConfig
    layout: RunLayout
    manifest: Manifest
    logger: logging.Logger

    def checkpoint_path(self, name: str) -> Path:
        return self.layout.checkpoint_path(name)

    def step_dir(self, name: str) -> Path:
        d = self.layout.step_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def input_frames(self, spec: StepSpec) -> list[pd.DataFrame]:
        frames = []
        for dep in spec.inputs:
            p = self.checkpoint_path(dep)
            if p.is_file():
                frames.append(pd.read_pickle(p))
        return frames

def _make_logger(layout: RunLayout) -> logging.Logger:
    logger = logging.getLogger(f"structurezyme.run.{layout.root.name}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(layout.log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler())
    return logger

class Runner:
    def __init__(self, config: RunConfig):
        # Enforce required paths at the real run boundary. Host defaults (if any)
        # must already have been applied by the caller (e.g. the CLI) before
        # constructing the Runner.
        config.validate_paths()
        self.config = config
        self.layout = RunLayout(run_dir(config.paths.output_root,
                                        config.runtime.user,
                                        config.runtime.run_id))
        self.layout.create()
        self.manifest = Manifest.load(self.layout.manifest_path)
        self.logger = _make_logger(self.layout)
        self.config.write(self.layout.config_path)

    def _existing_input_paths(self, spec: StepSpec):
        return [self.layout.checkpoint_path(d) for d in spec.inputs
                if self.layout.checkpoint_path(d).is_file()]

    def _should_skip(self, name: str, spec: StepSpec):
        if not self.config.is_enabled(name):
            return True, "SKIPPED_DISABLED"
        if name in self.config.runtime.force:
            return False, ""
        rec = self.manifest.get(name)
        out = self.layout.checkpoint_path(spec.output)
        if rec is None or not out.is_file():
            return False, ""
        cfg_hash = hash_obj(self.config.step_options(name))
        in_hash = hash_files(self._existing_input_paths(spec))
        if rec.status == "OK" and rec.config_hash == cfg_hash and rec.input_hash == in_hash:
            return True, "SKIPPED_CHECKPOINT"
        return False, ""

    def run(self, stop_after: str | None = None) -> None:
        """Execute the pipeline in dependency order.

        Args:
            stop_after: If given, halt the loop after this step has been
                processed (executed or skipped). Used by the CLI's ``step``
                command to run a single module without touching anything
                downstream of it. ``None`` runs the full pipeline.
        """
        if stop_after is not None and stop_after not in STEPS:
            raise KeyError(f"Unknown step {stop_after!r}; have {list(STEPS)}")
        for name in ordered_steps():
            spec = STEPS[name]
            skip, reason = self._should_skip(name, spec)
            if skip:
                self.logger.info(f"{reason}: {name}")
                if reason == "SKIPPED_CHECKPOINT":
                    # Preserve the existing OK record (status + timing metadata).
                    # Overwriting it would flip status away from "OK" and cause
                    # _should_skip to re-run the step on the next invocation,
                    # defeating resumability for every run after the second.
                    if name == stop_after:
                        break
                    continue
                # SKIPPED_DISABLED: no OK record to preserve; flag it explicitly.
                self.manifest.set(name, StepRecord(
                    status=reason,
                    pkl_path=str(self.layout.checkpoint_path(spec.output)),
                    config_hash=hash_obj(self.config.step_options(name)),
                    input_hash=hash_files(self._existing_input_paths(spec)),
                ))
                self.manifest.save(self.layout.manifest_path)
                if name == stop_after:
                    break
                continue

            self.logger.info(f"RUN: {name}")
            started = datetime.now().isoformat()
            t0 = time.time()
            self.manifest.set(name, StepRecord(status="RUNNING", started_at=started))
            self.manifest.save(self.layout.manifest_path)
            ctx = RunContext(self.config, self.layout, self.manifest, self.logger)
            try:
                df = spec.runner(ctx, spec)
                out = self.layout.checkpoint_path(spec.output)
                df.to_pickle(out)
                self.manifest.set(name, StepRecord(
                    status="OK",
                    pkl_path=str(out),
                    config_hash=hash_obj(self.config.step_options(name)),
                    input_hash=hash_files(self._existing_input_paths(spec)),
                    wall_time_s=time.time() - t0,
                    started_at=started,
                    finished_at=datetime.now().isoformat(),
                ))
                self.manifest.save(self.layout.manifest_path)
            except Exception as exc:
                self.manifest.set(name, StepRecord(
                    status="FAILED", started_at=started,
                    finished_at=datetime.now().isoformat(), error=repr(exc)))
                self.manifest.save(self.layout.manifest_path)
                self.logger.error(f"FAILED: {name}: {exc!r}")
                raise
            if name == stop_after:
                break
