# Task 02 — Install Miniforge to $HOME

**Depends on:** Task 01 (interactive session). **Owner unit:** standalone.

## Why

No conda/mamba/micromamba exists anywhere on PATH, and there is no `module` system. The repo's `environment.yml` is conda-based and pulls bioconda binaries. Miniforge (conda-forge default, ships `mamba`) is the lowest-risk manager. Install into `$HOME` (34 TB free).

## Commands

```bash
cd /tmp/kilo 2>/dev/null || cd /tmp
curl -L -o miniforge.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash miniforge.sh -b -p "$HOME/miniforge3"

# Activate for this shell
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda --version
mamba --version
```

Optional (persist for future shells):

```bash
"$HOME/miniforge3/bin/conda" init bash
# then: source ~/.bashrc   (or re-login)
```

## Notes

- Use `mamba` instead of `conda` for env solves — much faster on the heavy bioconda dependency set in Task 03.
- Do NOT use the system `/usr/bin/python3` (3.12). The repo targets 3.11.
- If the compute node has no internet egress, the `curl` will fail — see Task 03 "network" note; you may need to run the download on the login node first, or request a node/partition with outbound network.

## Verification

```bash
which conda mamba
conda info --base    # → $HOME/miniforge3
```

## Success criterion

`conda --version` and `mamba --version` both succeed and resolve under `$HOME/miniforge3`.
