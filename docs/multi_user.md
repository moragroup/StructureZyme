# Multi-user and Shared Hosts

Structurezyme is designed to run for multiple users on a shared host, with a
per-user output layout, host profiles for filling in shared paths, and file
locks to serialize writes to shared caches. It can also submit runs to Slurm.

## Per-user run layout

Every run is isolated under a per-user directory:

```
{output_root}/{user}/{run_id}/
```

`output_root` comes from the config (or a host profile, see below), `user` and
`run_id` come from the run's runtime config. This keeps each user's runs — and
each run's checkpoints, manifest, logs, and config — separate on shared storage.

## Host profiles (`hosts.yml`)

`hosts.yml` (shipped inside the `structurezyme` package) defines named profiles
that supply site-specific default paths. The `default` profile, for example:

```yaml
default:
  output_root: /mnt/labs/data/mora/structurezyme_runs
  boltz_cache_dir: /mnt/labs/data/mora/boltz_cache
  squidly_weights_dir: /mnt/labs/data/mora/squidly_weights
  placer_env_path: /mnt/labs/data/mora/software/PLACER/env
```

### Selecting a profile

The profile is chosen by, in order:

1. The `--host <name>` flag (on `run` and `submit`), if given.
2. Otherwise the `$STRUCTUREZYME_HOST` environment variable.
3. Otherwise the profile named `default`.

An unknown profile name raises an error listing the available profiles.

### How defaults are applied

`apply_host_defaults` only fills values you have **not** already set, so your
config always wins over the host profile:

- `output_root` — filled from the profile if unset (empty).
- `boltz_cache_dir` — filled from the profile if unset (empty).
- `squidly_weights_dir` — filled from the profile if `None`.
- `placer_env_path` — filled from the profile only if still at its built-in
  sentinel default (i.e. you have not overridden it).

## Shared caches and file locking

On a shared host, multiple users and jobs write into the same Boltz cache and
Squidly weights directories. To keep concurrent writes from colliding,
structurezyme guards those shared resources with an inter-process file lock via
`shared_lock(path)`.

`shared_lock` wraps a `filelock.FileLock` on `<path>.lock` (created next to the
guarded resource) and serializes access across processes:

```python
from structurezyme.locks import shared_lock

with shared_lock(boltz_cache_dir):
    # exclusive access to the shared cache here
    ...
```

By default the lock waits indefinitely for the resource; pass a `timeout`
(seconds) to fail fast instead.

## Submitting to Slurm

Use `submit` to render and submit a Slurm batch script for a run. The script is
generated from `structurezyme/templates/slurm.sbatch.j2` and, when submitted,
invokes `structurezyme run` on the compute node.

```bash
structurezyme submit --config config.yml \
  --host default \
  --job-name structurezyme \
  --partition gpu \
  --gpus 1 \
  --time 24:00:00 \
  --dry-run
```

Flags:

| Flag           | Default         | Meaning                                          |
| -------------- | --------------- | ------------------------------------------------ |
| `--config`     | (required)      | Config file passed to `structurezyme run`.       |
| `--host`       | `default`       | Host profile name for the run.                   |
| `--job-name`   | `structurezyme` | Slurm `--job-name`.                              |
| `--partition`  | `gpu`           | Slurm `--partition`.                             |
| `--gpus`       | `1`             | GPUs requested (`--gres=gpu:N`).                 |
| `--time`       | `24:00:00`      | Slurm `--time` limit.                            |
| `--dry-run`    | off             | Print the rendered script instead of submitting. |

With `--dry-run`, the rendered sbatch script is printed to stdout so you can
inspect it. Without `--dry-run`, the script is written to a temporary file and
submitted with `sbatch`.
