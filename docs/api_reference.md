# API Reference

This reference documents the public modules of the `structurezyme` package:
`config`, `registry`, `runner`, `cli`, and `hosts`. Signatures below match the
source and are intended as the contract for programmatic use.

## `structurezyme.config`

Configuration is modelled with Pydantic. A run is described by a single
`RunConfig` composed of three sub-models.

### `RunConfig`

```python
class RunConfig(BaseModel):
    paths: PathsConfig
    runtime: RuntimeConfig = RuntimeConfig()
    steps: StepsConfig = StepsConfig()
```

Methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `validate_paths` | `validate_paths() -> RunConfig` | Assert required paths are set; call **after** host defaults are applied. Raises `ValueError` if `output_root` or `boltz_cache_dir` is empty. Returns `self`. |
| `is_enabled` | `is_enabled(name: str) -> bool` | Whether the named step is enabled. |
| `step_options` | `step_options(name: str) -> dict` | The step's config as a dict (`model_dump()`), used for config hashing. |
| `write` | `write(path) -> None` | Serialise the config to YAML at `path` (creates parent dirs). |

### `load_config`

```python
def load_config(path) -> RunConfig
```

Read a YAML file and construct a `RunConfig`.

### `PathsConfig`

| Field | Type | Default |
|-------|------|---------|
| `output_root` | `str` | `""` |
| `boltz_cache_dir` | `str` | `""` |
| `squidly_weights_dir` | `str \| None` | `None` |
| `placer_env_path` | `str` | `"/mnt/labs/data/mora/software/PLACER/env"` |

`output_root` and `boltz_cache_dir` may be left empty at construction so a host
profile (see `hosts.apply_host_defaults`) can fill them; `validate_paths()`
enforces them at the run boundary.

### `RuntimeConfig`

| Field | Type | Default |
|-------|------|---------|
| `user` | `str` | current OS user (`getpass.getuser()`) |
| `run_id` | `str` | timestamp `YYYYMMDD-HHMMSS` |
| `num_threads` | `int` | `1` |
| `force` | `list[str]` | `[]` |

### `StepsConfig`

One `StepConfig` per pipeline step (`enabled: bool = True`, extra keys allowed).
Steps disabled by default: `vina`, `fastrelax`, `placer`.

## `structurezyme.registry`

The registry declares the pipeline graph as a static dictionary of specs.

### `StepSpec`

```python
@dataclass
class StepSpec:
    name: str
    depends_on: list[str] = []
    inputs: list[str] = []
    runner: Callable | None = None

    @property
    def output(self) -> str:  # equals name
        ...
```

`runner` is populated at import time from `structurezyme.step_runners`.

### `STEPS`

`STEPS: dict[str, StepSpec]` — the 15 registered steps, in declaration order:

`squidly`, `chai`, `boltz`, `vina`, `docking_metrics`, `prepare_files`,
`fastrelax`, `superimpose`, `protein_rmsd`, `ligand_rmsd`, `geometric_filter`,
`fpocket`, `ligand_sasa`, `plip`, `placer`.

Each step's `depends_on`/`inputs` wire the dependency chain. `fastrelax` is
optional but, when enabled, is declared as a dependency of `superimpose` to
enforce ordering; a disabled `fastrelax` is simply skipped at run time.

### `ordered_steps`

```python
def ordered_steps() -> list[str]
```

Return step names in a valid topological order (raises `ValueError` on a cycle).

## `structurezyme.runner`

### `Runner`

```python
class Runner:
    def __init__(self, config: RunConfig): ...
    def run(self, stop_after: str | None = None) -> None: ...
```

Construction calls `config.validate_paths()`, so host defaults must already be
applied by the caller. The constructor creates the run layout, loads (or starts)
the manifest, sets up logging, and writes the resolved config into the run dir.

`run()` executes steps in `ordered_steps()` order. For each step it decides
whether to skip:

- `SKIPPED_DISABLED` — the step is disabled in config.
- `SKIPPED_CHECKPOINT` — an `OK` manifest record exists, the output pickle is
  present, and both the config hash and input hash still match. The existing
  `OK` record is preserved (not overwritten) to keep resumability working across
  repeated runs.
- Otherwise the step runs; results are pickled to the checkpoint path and an
  `OK` record (with timing) is written. On exception a `FAILED` record is
  written and the error re-raised.

**`stop_after` semantics:** if `stop_after` is a step name, the loop halts after
that step has been *processed* (executed or skipped); it must be a valid key in
`STEPS` or `KeyError` is raised. `None` (default) runs the full pipeline. This
backs the CLI `step` command's single-module behaviour.

### `RunContext`

```python
@dataclass
class RunContext:
    config: RunConfig
    layout: RunLayout
    manifest: Manifest
    logger: logging.Logger
```

Passed to each step's runner. Helpers: `checkpoint_path(name)`,
`step_dir(name)` (created on access), and `input_frames(spec)` which loads the
pickled DataFrames for a spec's declared inputs.

## `structurezyme.cli`

Entry point:

```python
def main(argv=None) -> int
```

Parses `argv` (defaults to `sys.argv`) and dispatches to a subcommand handler,
returning its exit code.

### Subcommands

| Command | Key arguments | Action |
|---------|---------------|--------|
| `init` | `--output` (required) | Write a template `RunConfig` (empty paths) to the given file. |
| `run` | `--config` (required), `--host`, `--force` | Load config, apply host defaults, optionally set forced steps (comma-separated), run the full pipeline. |
| `resume` | `--run-dir` (required) | Reload the config saved in the run dir and continue the run. |
| `step` | `name` (positional), `--run-dir` (required), `--continue` | Force-run the named step. Default stops after it (`stop_after=name`); `--continue` also runs downstream steps, recomputing those whose inputs changed. |
| `status` | `--run-dir` (required) | Print each step's status and wall time from the manifest. Returns `1` if no manifest exists. |
| `submit` | `--config` (required), `--host`, `--job-name`, `--partition`, `--gpus`, `--time`, `--dry-run` | Render a Slurm `sbatch` script and submit it (`--dry-run` prints it instead). |

### `render_sbatch`

```python
def render_sbatch(config_path, host, job_name, partition, gpus, time_limit) -> str
```

Render the Slurm submission script from the packaged
`structurezyme.templates/slurm.sbatch.j2` template.

## `structurezyme.hosts`

Host profiles live in the packaged `structurezyme/hosts.yml` and supply
site-specific path defaults.

### `load_host_profile`

```python
def load_host_profile(name: str | None = None) -> dict
```

Return the named profile. If `name` is `None`, falls back to the
`STRUCTUREZYME_HOST` environment variable, then `"default"`. Raises `KeyError`
for an unknown profile.

### `apply_host_defaults`

```python
def apply_host_defaults(cfg: RunConfig, name: str | None = None) -> RunConfig
```

Fill unset path fields on `cfg.paths` from the selected profile: `output_root`
and `boltz_cache_dir` only if empty, `squidly_weights_dir` only if `None`, and
`placer_env_path` only if it is still the built-in sentinel default. Returns the
mutated `cfg`.
