# Resume and Checkpoints

Structurezyme runs are resumable. Each step writes a pickled result to
`checkpoints/<step>.pkl` and records its outcome in `manifest.json`. On the next
invocation the runner uses those artifacts to skip work that is already done and
still valid, so an interrupted or partially-failed run can be picked up where it
left off instead of starting over.

## Run directory layout

A run lives under `{output_root}/{user}/{run_id}/` and contains:

```
{output_root}/{user}/{run_id}/
├── config.yml            # the resolved config, rewritten on every run
├── manifest.json         # per-step status and provenance (see below)
├── logs/
│   └── structurezyme.log # run log
├── checkpoints/
│   └── <step>.pkl        # pickled output of each step
└── <step>/               # per-step working directories (created on demand)
```

## Manifest anatomy

`manifest.json` maps each step name to a record. The fields of a step record
are:

| Field         | Meaning                                                      |
| ------------- | ------------------------------------------------------------ |
| `status`      | One of the statuses below (`PENDING` before first run).      |
| `pkl_path`    | Absolute path to the step's `checkpoints/<step>.pkl` output. |
| `input_hash`  | Hash of the existing input checkpoint files for the step.    |
| `config_hash` | Hash of the step's resolved config options.                  |
| `wall_time_s` | Wall-clock seconds the step took (set on success).           |
| `mem_mb`      | Peak memory in MB, when recorded.                            |
| `started_at`  | ISO timestamp when the step started.                         |
| `finished_at` | ISO timestamp when the step finished.                        |
| `error`       | `repr()` of the exception when the step failed.              |

### Statuses

- `RUNNING` — the step is executing (written before the step body runs).
- `OK` — the step completed and its output pickle was written.
- `FAILED` — the step raised; `error` holds the exception repr. The run aborts.
- `SKIPPED_CHECKPOINT` — skipped because a valid checkpoint already exists.
- `SKIPPED_DISABLED` — skipped because the step is disabled in config.

Note: `SKIPPED_CHECKPOINT` is a log-only signal — the runner intentionally does
**not** overwrite the existing `OK` record when it skips via checkpoint, so that
resumability keeps working on every subsequent run. Only `SKIPPED_DISABLED` is
written into the manifest as its own status.

## The skip rule

When deciding whether to run a step, the runner applies these checks in order:

1. If the step is **not enabled** in config → skip, status `SKIPPED_DISABLED`.
2. Else if the step is in `runtime.force` → **run** it.
3. Else if there is no manifest record or the output pickle is missing → **run**.
4. Else if the recorded status is `OK` **and** the config hash **and** the input
   hash both match → skip, status `SKIPPED_CHECKPOINT`.
5. Otherwise → **run** (config or inputs changed).

In short: an enabled step is skipped only when it has a valid `OK` checkpoint
whose config and inputs are unchanged.

## Forcing steps to re-run

Use `--force` on `run` to force one or more steps to execute regardless of their
existing checkpoints. Pass a comma-separated list:

```bash
structurezyme run --config config.yml --force stepA,stepB
```

This re-runs `stepA` and `stepB` (and any downstream step whose inputs change as
a result), while other completed steps are still skipped via their checkpoints.

## Re-running a single step

To re-run one module in an existing run directory, use `step`:

```bash
structurezyme step <name> --run-dir {output_root}/{user}/{run_id}
```

This reloads `config.yml` from the run directory, forces `<name>`, and by
default **stops after that step** — nothing downstream is touched. To let the
rest of the pipeline follow (recomputing any downstream step whose inputs
changed), add `--continue`:

```bash
structurezyme step <name> --run-dir {run_dir} --continue
```

### Caveat: disabled steps cannot be force-run

`step <name>` adds `<name>` to `runtime.force`, but the skip rule checks
`enabled` **before** it checks `force` (check 1 above wins over check 2). So if
`<name>` is set to `enabled: false` in `config.yml`, `step <name>` will still
skip it as `SKIPPED_DISABLED` and it will not run. To force-run such a step you
must first enable it in the config, then run `step <name>` (or `resume`).

## Corruption recovery

If a checkpoint pickle is corrupt or was written by an aborted run, delete the
offending file and resume. Because the output pickle is now missing, the skip
rule (check 3) forces that step to re-run, and everything downstream of it
recomputes as needed:

```bash
rm {run_dir}/checkpoints/<step>.pkl
structurezyme resume --run-dir {run_dir}
```

`resume` reloads `config.yml` from the run directory and re-runs the full
pipeline; completed steps with valid checkpoints are skipped, so only the
deleted step and its dependents are recomputed.
