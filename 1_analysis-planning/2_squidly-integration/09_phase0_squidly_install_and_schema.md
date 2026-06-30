# 9. Phase 0 - Install `squidly` CLI and Discover Pickle Schema

**This is the first work item. Nothing in Phases A-F can be reliably planned until this
is done.** Estimated time: 30-60 minutes.

## Why this comes first

Verified facts about `enzymetk.ActiveSitePred` (from
`/mnt/storage01/home/lherrmann/envs/filterzyme/lib/python3.11/site-packages/enzymetk/predict_catalyticsite_step.py:45-59`):

- It does **not** load a model in Python. It shells out to a `squidly` CLI:
  `squidly run <input.fasta> <esm2_model> <tmp_dir> [args...]`.
- It reads `<tmp_dir>/squidly_ensemble.pkl` and returns whatever the CLI wrote.
- The `squidly` binary is **not installed** on this machine. Verified by:
  - `which squidly` -> not found
  - no `bin/squidly` under `envs/filterzyme/` or `envs/enzymetk/`
- This means the catalytic-residue path has **never executed** end-to-end on this host;
  the README example sets `skip_catalytic_residue_prediction=True` to dodge it.

We need to fix this before any other work because:

- The target column contract in `03_target_architecture.md` depends on what columns
  the CLI writes into `squidly_ensemble.pkl`.
- The threshold mechanism depends on what CLI flags `squidly run` accepts.
- The model-size kwarg `esm2_model` only matters if the CLI honours it.

## Step 0.1 - Find the upstream `squidly` source

Likely candidates to check (have not been verified):

- PyPI: `pip search squidly` (or browse pypi.org)
- GitHub: search for `Squidly` ESM2 LSTM catalytic residue predictor
- The `enzymetk` README at https://github.com/arianemora/enzyme-tk/ (enzymetk is by
  Ariane Mora; Squidly may be from the same group)
- Ask the PI for the canonical source

Record the answer in `07_risks_and_open_questions.md` Q-1.

## Step 0.2 - Install `squidly` into the `filterzyme` conda env

```
conda activate filterzyme
pip install squidly                         # if it is on PyPI
# OR
pip install git+https://<URL>/Squidly.git   # if it is git-only
```

Verify:

```
which squidly
squidly --help
squidly run --help
```

Record the full `squidly run --help` output in this file (append a section "0.2 output").

## Step 0.3 - Run `squidly` once on a tiny fasta

Create `/tmp/squidly_smoke.fasta`:

```
>test_serine_hydrolase
MKAILVVLLFTLVASA...  # ~50-150 aa, any sequence with a real catalytic residue
```

Then:

```
mkdir -p /tmp/squidly_out
squidly run /tmp/squidly_smoke.fasta esm2_t36_3B_UR50D /tmp/squidly_out
ls /tmp/squidly_out
```

Expected: a file `squidly_ensemble.pkl` exists.

## Step 0.4 - Discover the pickle schema

```python
import pandas as pd
df = pd.read_pickle('/tmp/squidly_out/squidly_ensemble.pkl')
print(df.columns.tolist())
print(df.dtypes)
print(df.head(1).to_dict())
```

Record the output here as a "schema discovery result" section. Specifically note whether
the following columns exist:

| Column we want | Present? | If not, can we derive it? |
|----------------|----------|---------------------------|
| `label` / `Entry` (sequence id) | ? | required |
| `Squidly_Ensemble_Residues` (or similar) - residue list | ? | required |
| per-residue scores / probabilities | ? | nice-to-have |
| threshold used | ? | nice-to-have |

`pipeline_v2.py:104` currently reads `df_cat_res.label` and
`df_cat_res.Squidly_Ensemble_Residues`. If those two columns exist, the existing
pipeline_v2 code already works against the CLI's pkl with no rename layer.

## Step 0.5 - Decide the column contract

Based on Step 0.4, finalize `03_target_architecture.md`'s column contract:

- If scores are in the pkl -> keep `squidly_scores` in the contract.
- If not -> drop `squidly_scores`; only `catalytic_residues` (and optionally
  `squidly_threshold` recorded by our wrapper) survive.

## Step 0.6 - Document install steps for the lab sanity run

Once `squidly` is working, add a new file `3_setup-sanity-check/10_squidly_install.md`
that records the exact install command, the verified `squidly --version`, and the smoke
test output. This becomes part of repo provenance.

## Exit criteria

- [ ] `which squidly` resolves inside `filterzyme` conda env.
- [ ] `squidly run --help` output captured here.
- [ ] `squidly_ensemble.pkl` smoke test passes on a 1-sequence fasta.
- [ ] Pickle schema documented in this file.
- [ ] `03_target_architecture.md` column contract reconciled with reality (or a note
      added saying it is unchanged).

Only when all five boxes are checked does Phase A in `05_implementation_steps.md`
begin.

---

## 0.2 output (fill in)

_paste `squidly --help` and `squidly run --help` here once installed_

## 0.4 output (fill in)

_paste `df.columns.tolist()` and `df.dtypes` here once the smoke test runs_
