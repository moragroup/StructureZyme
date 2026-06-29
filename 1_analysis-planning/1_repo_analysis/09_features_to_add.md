# 9. Key Features to Add

These are user-visible capabilities, not internal refactors. Each entry is described as a capability, an implementation sketch, and a success criterion.

## F-1: Multi-GPU / distributed execution

**Capability.** A run with `--gpus 0,1,2,3` partitions entries across four CUDA devices. A later iteration adds Dask/Ray for multi-node SLURM execution.

**Implementation.**

- Add `gpu_devices: list[int]` to `PipelineConfig`.
- In the Chai/Boltz/DiffDock workers, set `CUDA_VISIBLE_DEVICES = str(gpu_devices[worker_id % len(gpu_devices)])` before importing torch.
- Use `ProcessPoolExecutor` so each worker gets a fresh interpreter (required because torch caches the device at import time).
- For the SLURM phase: integrate `dask-jobqueue` or `ray[default]` with one job per entry.

**Success.** A 20-entry sweep on a 4-GPU box finishes in roughly 1/4 the wall-clock time of a 1-GPU sweep, within Chai/Boltz throughput.

## F-2: Unified cofactor handling

**Capability.** A single cofactor schema understood by Chai, Boltz, Vina, and the geometric filters. Optional multi-cofactor (e.g. NADPH + metal).

**Implementation.**

- Pick one sentinel (`None`) for "no cofactor" and apply it consistently. Currently `pipeline.py:127-128` uses `''` for Chai and `pipeline.py:137-138` uses `None` for Boltz â unify.
- Allow `cofactor_smiles` to be `str | list[str]` and document the list semantics (multiple cofactors bind simultaneously).
- The geometric-filter family (`geometric_filtering_cofactor_MCS.py:435`) already accepts a single cofactor; extend to a list and update the distance/SMARTS loops.
- Add an integration test on a heme-binding example.

**Success.** `examples/DEHP-MEHP-with-cofactor.pkl` runs end-to-end and produces a `cofactor_substrate_distance` column.

## F-3: YAML config file

**Capability.** A single `config.yaml` (Pydantic-validated) drives the whole pipeline. CLI is `filterzyme run --config path/to/config.yaml --input path/to/df.pkl`.

**Implementation.** See item 4 in `07_top10_improvements.md`. Ship a documented `examples/config.example.yaml` with every option commented.

**Success.** No parameter is set in Python source. A new substrate can be processed by editing only the YAML and the input pickle.

## F-4: Checkpoint / resume with stage hashing

**Capability.** Re-running with the same `--config` and `--input` skips any stage whose output pickle exists and is fresher than its inputs. A `--force` flag bypasses the cache; `--force-stage vina` re-runs only Vina.

**Implementation.**

- Each stage computes `key = sha256(config_blob + input_pickle_path + input_mtime)` and writes to `<output_dir>/<stage>__<key>.pkl`.
- A `<output_dir>/manifest.json` records `{stage_name: key}` for the latest successful run.
- On startup, read the manifest and short-circuit any stage whose key matches.

**Success.** Killing a run after Vina and restarting jumps straight to Superimposition in <5 s.

## F-5: Per-entry error report

**Capability.** A `<output_dir>/errors.csv` with columns `Entry, stage, error_class, error_message, traceback_path` for every entry that failed at any stage. The run continues for the remaining entries.

**Implementation.**

- Wrap each per-entry call in `try/except`. On failure, append a row to `errors.csv` and write the traceback to `<output_dir>/tracebacks/<Entry>__<stage>.txt`.
- Mark the failed entry's row in the per-stage DataFrame with `NaN` so downstream stages skip it cleanly.
- At the end of the run, print a summary: `12/120 entries failed; see errors.csv`.

**Success.** A run with one deliberately bad entry (e.g. invalid SMILES) completes the other entries and produces an actionable `errors.csv`.

## F-6: Improved logging

**Capability.** Structured logs, one log file per run, per-worker tagging, JSON-output mode for downstream parsing. Plays well with `tqdm` progress bars.

**Implementation.**

- Configure logging in `filterzyme/__init__.py` with a `RotatingFileHandler` writing to `<output_dir>/filterzyme.log`.
- Inject a `contextvars.ContextVar("worker_id")` into every log record via a filter.
- Optional `--log-format json` flag emits JSON-Lines for ingestion into ELK/Loki.
- Replace `print` with `logger.info` outside `__main__` blocks (CQ-9 cleanup).

**Success.** A multi-worker run produces interleaved-yet-traceable log output; switching to `--log-format json` produces valid JSONL.

## F-7: CLI entry point

**Capability.** `filterzyme run --config config.yaml --input df.pkl --output out/` works after `pip install filterzyme`.

**Implementation.**

- Add `entry_points={"console_scripts": ["filterzyme = filterzyme.cli:main"]}` to `setup.py`.
- Implement `filterzyme/cli.py` with `click` or `typer` subcommands: `run`, `validate`, `resume`, `list-stages`.

**Success.** `which filterzyme` resolves after install; `filterzyme --help` lists subcommands.

## F-8: Reproducibility manifest

**Capability.** Every run writes a `<output_dir>/run_manifest.json` capturing: `filterzyme` version, all upstream-tool versions (Chai, Boltz, docko, enzymetk), the config blob, the input pickle SHA-256, the git commit (if applicable), and the host hostname + GPU model. Sufficient to reproduce a run.

**Implementation.** ~50 LOC in `filterzyme/utils/manifest.py`. Call from `Pipeline.run` at the very start and end.

**Success.** Two runs with identical manifests produce numerically identical outputs (modulo Chai/Boltz stochasticity, which itself is captured in the manifest as the RNG seed).

## F-9: Web-facing report (stretch)

**Capability.** A `filterzyme report <output_dir>` command renders an HTML report with per-entry pose images (RDKit), score tables, and PLIP interaction diagrams. Useful for sharing results with wet-lab collaborators.

**Implementation.** Jinja2 template, RDKit `MolToImage`, embedded base64 PNGs. ~200 LOC.

**Success.** `<output_dir>/report.html` opens in a browser and shows every entry with its best pose.

**Phase.** Optional, Phase 5 or later.
