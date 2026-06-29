# Task 01 — Get an interactive RTX6000 Slurm session

**Depends on:** nothing. **Owner unit:** standalone.

## Why

The login node (`login02`) has no GPU and no conda. Chai-1, Boltz-2, and Squidly (ESM2) need a GPU. All install + run work happens inside an interactive compute session on the `gpu` partition.

## Verified facts

- Scheduler: Slurm (`sbatch`, `srun`, `salloc`, `sinfo` all at `/usr/bin/`).
- Account: `mora`. QOS: `debug`, `normal`, `spot`.
- `gpu` partition: RTX6000, `gpu:rtx6000:8(S:0-1)`, 24 GB VRAM/card, 4-day TIMELIMIT.

## Commands

Request one RTX6000 GPU, 8 CPUs, 48 GB RAM, 8 hours:

```bash
salloc -p gpu --gres=gpu:rtx6000:1 -A mora -c 8 --mem=48G -t 08:00:00
```

If `salloc` drops you into the allocation but not onto the node, or you prefer an interactive shell directly:

```bash
srun -p gpu --gres=gpu:rtx6000:1 -A mora -c 8 --mem=48G -t 08:00:00 --pty bash -l
```

Notes:
- `-A mora` is the verified account. If Slurm rejects it, run `sacctmgr -nP show assoc user=$USER format=Account,QOS` and use the listed account.
- Default QOS `normal` should be fine; only add `--qos=debug` for a short test or `--qos=spot` on the `spot` partition.
- If the `gpu` partition is full, fall back to `-p spot --gres=gpu:rtx6000:1` (preemptible) or `-p h100 --gres=gpu:h100nvl:1`.

## Verification (must pass before moving on)

```bash
hostname                 # should NOT be login02 — you are on a compute node
nvidia-smi               # should list one RTX6000, ~24 GB, and a driver/CUDA version
echo "$CUDA_VISIBLE_DEVICES"
nproc                    # should reflect the -c value
```

**Record the CUDA version reported by `nvidia-smi`** (top-right of the table). Task 04 needs it to pick the matching torch wheel (`cu118` / `cu121` / `cu124`).

## Success criterion

`nvidia-smi` shows an RTX6000 on a non-login host. Note the CUDA driver version for Task 04.
