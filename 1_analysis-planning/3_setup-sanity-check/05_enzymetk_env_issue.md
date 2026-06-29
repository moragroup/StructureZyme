# Task 05 — Resolve the enzymetk `env_name` / tool-env requirement (BIGGEST RISK)

**Depends on:** Task 04. **Owner unit:** standalone but blocks Task 07's Docking stage.

## The problem

`filterzyme/pipeline.py` calls the enzymetk wrappers WITHOUT an `env_name` argument:

- line 86: `pred_in << ActiveSitePred('Entry', 'Sequence')`           (Squidly)
- line 134: `Chai('Entry', 'Sequence', 'substrate_smiles', 'cofactor_smiles', chai_dir, self.num_threads)`
- line 145: `Boltz('Entry', 'Sequence', 'substrate_smiles', 'cofactor_smiles', boltz_dir, self.num_threads)`

Per the enzymetk design, many Step wrappers shell out to their external tool **inside a conda env whose name defaults to `enzymetk`** (each Step accepts `env_name=` to override). enzymetk also installs tools per-module via `conda_envs` scripts in its repo and a wiki: https://github.com/moragroup/enzyme-tk/wiki/Installations

So the Docking stage may try to `conda run -n enzymetk ...` (or similar) and fail if that env does not exist or lacks chai/boltz/esm.

## Step 1 — Determine the actual behaviour of THIS enzymetk version

Inspect the installed wrappers (do not guess — versions differ):

```bash
conda activate filterpipeline
python - <<'PY'
import inspect, enzymetk
import enzymetk.dock_chai_step as c
import enzymetk.dock_boltz_step as b
import enzymetk.predict_catalyticsite_step as a
for mod,name in [(c,"Chai"),(b,"Boltz"),(a,"ActiveSitePred")]:
    print("="*60, name, mod.__file__)
    src = inspect.getsource(getattr(mod, name).__init__)
    print(src)
PY
```

Look for: a default `env_name` (e.g. `"enzymetk"`), any `conda run -n` / `subprocess` calls, or whether it imports and runs chai/boltz/torch IN-PROCESS.

Two outcomes:

- **(A) In-process** (it imports chai_lab/boltz/torch directly): no extra env needed. Proceed to Task 06/07.
- **(B) Subprocess into a named conda env**: you must create that env with the tools, OR pass `env_name=` pointing at `filterpipeline`. Since the pipeline does NOT pass `env_name`, the cleanest sanity-check options are below.

## Step 2 (only if outcome B) — make the tool env reachable

Option B1 — create the env enzymetk expects (likely `enzymetk`) using its own install scripts/wiki, installing chai-lab, boltz, ESM/torch, and pointing Squidly at the weights in `filterzyme/squidly_final_models/`.

Option B2 — minimal patch for sanity check ONLY (note it in the report, do not commit as a fix): temporarily pass `env_name='filterpipeline'` in `pipeline.py` lines 86/134/145, so the wrappers reuse the already-built env instead of a missing `enzymetk` env. This avoids building a second heavyweight env just to test.

## Squidly specifics

`ActiveSitePred` (Squidly) needs the ESM2 embeddings (downloaded at runtime, multi-GB) plus the LSTM heads already in the repo: `filterzyme/squidly_final_models/Squidly_LSTM_3B.pth` / `Squidly_LSTM_15B.pth`. The benchmark scripts set `skip_catalytic_residue_prediction=True` and pass `squidly_dir`; for the example run you can also skip Squidly (provide `vina_residues`) to isolate docking from active-site prediction — see Task 07.

## Success criterion

You can state definitively: does enzymetk Chai/Boltz/ActiveSitePred run in-process or via a named conda env? If named-env, the env exists (or `env_name` workaround applied) so Task 07's Docking stage can attempt to launch.
