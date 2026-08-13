# FastRelax + PLACER Inclusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FastRelax run from the structurezyme env by calling the existing RosettaFastRelax venv via subprocess (mirroring PLACER), enable PLACER, and prove the full 15-step DAG end-to-end on GPU on the fresh handoff env.

**Architecture:** The pyrosetta-touching body of `FastRelax._relax_one` moves to a standalone worker script (`_fastrelax_worker.py`) that imports only stdlib + pyrosetta and runs under the RosettaFastRelax venv python. `_relax_one` keeps its exact signature `(self, pdb_path, substrate_smiles=None, cofactor_smiles=None) -> (str, float)` but now writes a JSON spec, invokes the worker via `subprocess.run`, and parses the result. All pure-Python prep (ligand registration, `.params` generation, PDB prep, ranking) stays in-process. PLACER is already subprocess-correct; it just needs enabling. Verification switches the smoke to the modular runner (`structurezyme run --config ... --host default`) which already wires all 15 steps including fastrelax + placer.

**Tech Stack:** Python 3.11 (structurezyme env), pyrosetta (external venv, py3.12), pytest, SLURM/sbatch, YAML run configs, pandas.

## Global Constraints

- No pyrosetta or PLACER added to `environment.yml` (licensed / heavy / external).
- FastRelax venv default: `/mnt/labs/data/mora/software/RosettaFastRelax/env`; override via env var `FASTRELAX_ENV` (env dir; python is `<env>/bin/python`).
- PLACER install default: `/mnt/labs/data/mora/software/PLACER/env` (host-profile `placer_env_path`).
- `_relax_one` public signature MUST stay `(self, pdb_path, substrate_smiles=None, cofactor_smiles=None) -> tuple[str, float]` (existing tests monkeypatch it by this signature: `tests/test_fastrelax_step.py:205`).
- Per-pose failure tolerance in `execute()` MUST be preserved: a failing `_relax_one` is logged and the original path retained (`fastrelax_step.py:625-636`).
- Behavior parity: `-extra_res_fa` params list, `.for_pose.pdb` prep, ligand_focused vs full movemap, coordinate constraints, output filename `<stem>_relaxed.pdb`, and returned score must be identical to the current inline code (`fastrelax_step.py:504-577`).
- Fresh handoff env for GPU verification: `/mnt/storage01/home/lherrmann/envs/structurezyme` (pass via `STRUCTUREZYME_ENV`).
- SLURM `--output/--error` MUST be on shared storage; smoke uses gpu partition.
- Full suite baseline that must stay green: 242 passed / 5 skipped / 0 failed.
- Local commits only; no push/merge/rebase without explicit user approval.

## File Structure

- `structurezyme/steps/_fastrelax_worker.py` (NEW) — standalone pyrosetta worker; stdin/argv JSON spec in, JSON result out; no structurezyme imports.
- `structurezyme/steps/fastrelax_step.py` (MODIFY) — replace the pyrosetta body of `_relax_one` with a subprocess call; add venv-python resolution (`FASTRELAX_ENV`) + worker-path lookup + JSON spec builder + result parser; change `_pyrosetta_available()` to a venv-availability check.
- `tests/test_fastrelax_subprocess.py` (NEW) — unit tests for spec builder, venv resolution/override, result parsing (no pyrosetta needed; mock `subprocess.run`).
- `tests/test_fastrelax_preserves_ligand_atoms.py`, `tests/test_fastrelax_boltz_atom_names.py` (MODIFY) — update the skip guard to the new venv-availability helper.
- `tools/smoke/run_full_dag_smoke.sbatch` (NEW) — sbatch that runs the modular full-DAG on the fresh env and validates fastrelax + placer + final pickle. (Models `tools/gpu_run/run_allmodules.sbatch`.)
- `docs/getting_started.md`, `docs/known-issues.md` (MODIFY) — document both shared installs, `FASTRELAX_ENV`, enable steps, and the FastRelax subprocess design.

---

### Task 1: FastRelax pyrosetta worker script

**Files:**
- Create: `structurezyme/steps/_fastrelax_worker.py`
- Test: covered indirectly by Task 3 (spec shape) + the GPU smoke (Task 6). No unit test here (needs real pyrosetta).

**Interfaces:**
- Consumes: a JSON spec file path as `sys.argv[1]` with keys
  `prepared_pdb` (str), `extra_res_fa` (list[str]), `mode` ("ligand_focused"|"full"),
  `ligand_resname` (str|null), `shell_radius` (float), `constraint_weight` (float),
  `scorefunction` (str), `out_pdb` (str).
- Produces: prints exactly one JSON object as the LAST line of stdout:
  `{"relaxed_path": <str>, "score": <float>}` on success; on failure prints
  `{"error": <str>}` and exits 1.

- [ ] **Step 1: Write the worker script**

Create `structurezyme/steps/_fastrelax_worker.py`. This is a direct port of the
inline pyrosetta body at `fastrelax_step.py:488-577`, parameterized by the JSON
spec instead of `self`. It imports ONLY stdlib + pyrosetta.

```python
"""Standalone PyRosetta FastRelax worker.

Run under the RosettaFastRelax venv python (py3.12), NOT the structurezyme
env. Imports only stdlib + pyrosetta so it never touches structurezyme code.

Usage:
    python _fastrelax_worker.py <spec.json>

Reads a JSON spec (see keys below), runs FastRelax, writes the relaxed PDB to
`out_pdb`, and prints `{"relaxed_path": ..., "score": ...}` as the LAST line of
stdout. On any error prints `{"error": ...}` and exits 1.
"""
import json
import sys


def _run(spec: dict) -> dict:
    import pyrosetta

    extra_res_fa = " ".join(spec["extra_res_fa"])
    init_opts = "-mute all"
    if extra_res_fa:
        init_opts = f"{init_opts} -extra_res_fa {extra_res_fa}"
    pyrosetta.init(extra_options=init_opts, silent=True)

    pose = pyrosetta.pose_from_pdb(spec["prepared_pdb"])
    mode = spec["mode"]
    ligand_resname = spec.get("ligand_resname")
    shell_radius = spec["shell_radius"]
    constraint_weight = spec["constraint_weight"]
    scorefunction = spec["scorefunction"]

    movemap = None
    movemap_factory = None
    if mode == "ligand_focused":
        from pyrosetta.rosetta.core.select.residue_selector import (
            NeighborhoodResidueSelector,
            ResidueNameSelector,
        )
        from pyrosetta.rosetta.core.select.movemap import (
            MoveMapFactory,
            move_map_action,
        )
        from pyrosetta.rosetta.protocols.constraint_generator import (
            AddConstraints,
            CoordinateConstraintGenerator,
        )

        if ligand_resname is None:
            raise RuntimeError(
                "mode='ligand_focused' requires a ligand_resname; got None."
            )
        ligand_sel = ResidueNameSelector()
        ligand_sel.set_residue_name3(ligand_resname)
        shell_sel = NeighborhoodResidueSelector(ligand_sel, shell_radius, True)

        movemap_factory = MoveMapFactory()
        movemap_factory.all_bb(False)
        movemap_factory.all_chi(False)
        movemap_factory.add_bb_action(move_map_action.mm_enable, shell_sel)
        movemap_factory.add_chi_action(move_map_action.mm_enable, shell_sel)

        coord_gen = CoordinateConstraintGenerator()
        coord_gen.set_residue_selector(shell_sel)
        coord_gen.set_sd(1.0 / max(constraint_weight, 1e-6))
        add_csts = AddConstraints()
        add_csts.add_generator(coord_gen)
        add_csts.apply(pose)
    elif mode == "full":
        movemap = pyrosetta.MoveMap()
        movemap.set_bb(True)
        movemap.set_chi(True)
    else:
        raise ValueError(f"Unknown FastRelax mode: {mode!r}")

    scorefxn = pyrosetta.create_score_function(scorefunction)
    if mode == "ligand_focused":
        from pyrosetta.rosetta.core.scoring import ScoreType
        scorefxn.set_weight(ScoreType.coordinate_constraint, constraint_weight)

    relax = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn)
    if movemap_factory is not None:
        relax.set_movemap_factory(movemap_factory)
    elif movemap is not None:
        relax.set_movemap(movemap)
    relax.apply(pose)

    pose.dump_pdb(spec["out_pdb"])
    return {"relaxed_path": spec["out_pdb"], "score": float(scorefxn(pose))}


def main() -> int:
    try:
        with open(sys.argv[1]) as fh:
            spec = json.load(fh)
        result = _run(spec)
    except Exception as e:  # noqa: BLE001 - surface any failure as JSON
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Sanity-check the file imports cleanly (no pyrosetta needed for parse)**

Run: `python -c "import ast; ast.parse(open('structurezyme/steps/_fastrelax_worker.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add structurezyme/steps/_fastrelax_worker.py
git commit -m "feat(fastrelax): add standalone pyrosetta FastRelax worker script"
```

---

### Task 2: Add venv resolution + worker helpers to FastRelax

**Files:**
- Modify: `structurezyme/steps/fastrelax_step.py`
- Test: `tests/test_fastrelax_subprocess.py` (created in Task 3)

**Interfaces:**
- Consumes: nothing new.
- Produces (module-level + method helpers used by Task 3's `_relax_one`):
  - `_resolve_fastrelax_python() -> Path` — returns the venv python, honoring
    `FASTRELAX_ENV`; raises `FileNotFoundError` if missing.
  - `_worker_path() -> Path` — absolute path to `_fastrelax_worker.py`.
  - `_parse_worker_stdout(stdout: str) -> tuple[str, float]` — parse the last JSON line; raise `RuntimeError` on error/missing keys.

  (The JSON spec dict is built inline inside `_relax_one` in Task 3, not as a
  separate method — it needs `self`'s params + the per-pose prepared paths.)

- [ ] **Step 1: Add imports and module-level resolver helpers**

At the top of `structurezyme/steps/fastrelax_step.py`, add `import json`,
`import os`, `import subprocess` to the existing imports. Then add these
module-level helpers (near `_pyrosetta_available`, `fastrelax_step.py:44`):

```python
_DEFAULT_FASTRELAX_ENV = "/mnt/labs/data/mora/software/RosettaFastRelax/env"


def _resolve_fastrelax_python() -> Path:
    """Return the RosettaFastRelax venv python.

    Honors env var ``FASTRELAX_ENV`` (an env *directory*; python is
    ``<env>/bin/python``); falls back to the shared install default.
    Raises FileNotFoundError with an actionable message if absent.
    """
    env_dir = os.environ.get("FASTRELAX_ENV", _DEFAULT_FASTRELAX_ENV)
    python = Path(env_dir) / "bin" / "python"
    try:
        exists = python.exists()
    except OSError:
        exists = False
    if not exists:
        raise FileNotFoundError(
            f"FastRelax pyrosetta env python not found at {python}. "
            f"Set FASTRELAX_ENV to the RosettaFastRelax env dir or install it "
            f"(see /mnt/labs/data/mora/software/RosettaFastRelax/)."
        )
    return python


def _worker_path() -> Path:
    """Absolute path to the standalone pyrosetta worker script."""
    return Path(__file__).with_name("_fastrelax_worker.py")


def _parse_worker_stdout(stdout: str) -> tuple[str, float]:
    """Parse the worker's last stdout line as JSON -> (relaxed_path, score).

    Raises RuntimeError if the line is missing, not JSON, carries an
    ``error`` key, or lacks the expected result keys.
    """
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("FastRelax worker produced no output")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"FastRelax worker stdout not JSON: {lines[-1]!r}"
        ) from e
    if "error" in payload:
        raise RuntimeError(f"FastRelax worker error: {payload['error']}")
    if "relaxed_path" not in payload or "score" not in payload:
        raise RuntimeError(
            f"FastRelax worker returned unexpected payload: {payload!r}"
        )
    return str(payload["relaxed_path"]), float(payload["score"])
```

- [ ] **Step 2: Change `_pyrosetta_available()` to a venv-availability check**

Replace the body of `_pyrosetta_available` (`fastrelax_step.py:44-54`). It must
no longer import pyrosetta in-process (it isn't in the env); instead it reports
whether the external venv is reachable, which is what the smoke tests now need.

```python
def _pyrosetta_available() -> bool:
    """True iff the external RosettaFastRelax venv python is reachable.

    Used to gate smoke tests that run real Rosetta via subprocess. Does NOT
    import pyrosetta in-process (pyrosetta lives in a separate venv, not the
    structurezyme env). Honors FASTRELAX_ENV.
    """
    try:
        _resolve_fastrelax_python()
        return True
    except FileNotFoundError:
        return False
```

- [ ] **Step 3: Verify import still works and the suite is unaffected so far**

Run: `python -c "from structurezyme.steps.fastrelax_step import _resolve_fastrelax_python, _worker_path, _parse_worker_stdout, _pyrosetta_available; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add structurezyme/steps/fastrelax_step.py
git commit -m "feat(fastrelax): add venv resolver, worker-path, stdout parser; venv-based _pyrosetta_available"
```

---

### Task 3: Refactor `_relax_one` to invoke the worker via subprocess

**Files:**
- Modify: `structurezyme/steps/fastrelax_step.py` (`_relax_one`, lines 457-580)

**Interfaces:**
- Consumes: `_resolve_fastrelax_python`, `_worker_path`, `_parse_worker_stdout`
  (Task 2); `self._prepare_pdb_for_pose`, `self._ligand_params` (unchanged).
- Produces: `_relax_one(self, pdb_path, substrate_smiles=None, cofactor_smiles=None) -> tuple[str, float]`
  — SAME signature as before, still returns `(relaxed_pdb_path, final_score)`.

- [ ] **Step 1: Replace the pyrosetta body of `_relax_one` with a subprocess call**

Replace the method body from the `try: import pyrosetta` block through the
`return (str(relaxed_path), float(scorefxn(pose)))` (`fastrelax_step.py:479-577`)
with the subprocess implementation below. KEEP the surrounding docstring and the
outer `except Exception ... raise` (lines 578-580) intact. The prep call
`self._prepare_pdb_for_pose(...)` still runs IN-PROCESS (it does RDKit work and
must stay in the structurezyme env); only the Rosetta relaxation is delegated.

```python
        env_python = _resolve_fastrelax_python()

        prepared_pdb, sub_resname, cof_resname = self._prepare_pdb_for_pose(
            pdb_path, substrate_smiles, cofactor_smiles,
        )
        # Ligand resname used by ligand_focused mode's neighborhood selector.
        # Prefer the substrate; fall back to the cofactor.
        ligand_resname_for_selector = sub_resname or cof_resname

        self.output_dir.mkdir(parents=True, exist_ok=True)
        relaxed_path = self.output_dir / f"{Path(pdb_path).stem}_relaxed.pdb"

        # Pass every registered params file to Rosetta via -extra_res_fa.
        # Without this Rosetta silently maps any unknown LIG residue to a
        # 29-atom generic template and corrupts the ligand chemistry.
        extra_res_fa = [str(lp.params_path) for lp in self._ligand_params.values()]
        spec = {
            "prepared_pdb": str(prepared_pdb),
            "extra_res_fa": extra_res_fa,
            "mode": self.mode,
            "ligand_resname": ligand_resname_for_selector,
            "shell_radius": self.shell_radius,
            "constraint_weight": self.constraint_weight,
            "scorefunction": self.scorefunction,
            "out_pdb": str(relaxed_path),
        }
        spec_path = self.output_dir / f"{Path(pdb_path).stem}_relax_spec.json"
        spec_path.write_text(json.dumps(spec))

        result = subprocess.run(
            [str(env_python), str(_worker_path()), str(spec_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FastRelax worker failed (rc={result.returncode}) for "
                f"{pdb_path}: {result.stdout[-500:]} {result.stderr[-500:]}"
            )
        relaxed_str, score = _parse_worker_stdout(result.stdout)
        return (relaxed_str, score)
```

- [ ] **Step 2: Remove the now-dead module-level pyrosetta-init guard**

The `_pyrosetta_initialized` global (`fastrelax_step.py:41`) and its use inside
`_relax_one` are gone (init now happens per-subprocess in the worker). Delete the
`_pyrosetta_initialized = False` line at module scope if it is no longer
referenced anywhere. Verify with:

Run: `grep -n "_pyrosetta_initialized" structurezyme/steps/fastrelax_step.py`
Expected: no output (all references removed).

- [ ] **Step 3: Verify import + full existing suite still green**

Run: `python -c "import structurezyme.steps.fastrelax_step; print('OK')"`
Expected: `OK`

Run: `pytest tests/test_fastrelax_step.py -q`
Expected: PASS (the pure-Python execute/validation tests; the mocked-`_relax_one`
failure-tolerance test at `tests/test_fastrelax_step.py:183` still passes because
`_relax_one`'s signature is unchanged).

- [ ] **Step 4: Commit**

```bash
git add structurezyme/steps/fastrelax_step.py
git commit -m "refactor(fastrelax): run pyrosetta relaxation via subprocess worker in the RosettaFastRelax venv"
```

---

### Task 4: Unit tests for the subprocess seam (no pyrosetta needed)

**Files:**
- Create: `tests/test_fastrelax_subprocess.py`

**Interfaces:**
- Consumes: `_resolve_fastrelax_python`, `_worker_path`, `_parse_worker_stdout`
  (Task 2); `FastRelax._relax_one` (Task 3), mocking `subprocess.run`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fastrelax_subprocess.py`:

```python
"""Unit tests for FastRelax's subprocess seam (no pyrosetta required).

These mock subprocess.run / the filesystem so they run in the plain
structurezyme env with no RosettaFastRelax venv present.
"""
import json
import subprocess
from pathlib import Path

import pytest

from structurezyme.steps import fastrelax_step
from structurezyme.steps.fastrelax_step import (
    FastRelax,
    _parse_worker_stdout,
    _resolve_fastrelax_python,
    _worker_path,
)


def test_worker_path_points_at_worker_file():
    p = _worker_path()
    assert p.name == "_fastrelax_worker.py"
    assert p.exists()


def test_resolve_python_honors_env_override(tmp_path, monkeypatch):
    env_dir = tmp_path / "myenv"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    monkeypatch.setenv("FASTRELAX_ENV", str(env_dir))
    assert _resolve_fastrelax_python() == env_dir / "bin" / "python"


def test_resolve_python_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTRELAX_ENV", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="pyrosetta env python not found"):
        _resolve_fastrelax_python()


def test_parse_stdout_happy_path():
    out = 'noise line\n{"relaxed_path": "/x/a_relaxed.pdb", "score": -12.5}\n'
    path, score = _parse_worker_stdout(out)
    assert path == "/x/a_relaxed.pdb"
    assert score == -12.5


def test_parse_stdout_error_key_raises():
    with pytest.raises(RuntimeError, match="worker error: boom"):
        _parse_worker_stdout('{"error": "boom"}')


def test_parse_stdout_non_json_raises():
    with pytest.raises(RuntimeError, match="not JSON"):
        _parse_worker_stdout("not json at all")


def test_parse_stdout_empty_raises():
    with pytest.raises(RuntimeError, match="no output"):
        _parse_worker_stdout("   \n  \n")
```

- [ ] **Step 2: Run to verify they pass (helpers already exist from Tasks 2-3)**

Run: `pytest tests/test_fastrelax_subprocess.py -q`
Expected: PASS (7 tests).

- [ ] **Step 3: Add a `_relax_one` end-to-end mock test (subprocess mocked)**

Append to `tests/test_fastrelax_subprocess.py`:

```python
def test_relax_one_invokes_worker_and_parses(tmp_path, monkeypatch):
    """_relax_one builds a spec, calls the worker, and returns its result --
    with subprocess.run and _prepare_pdb_for_pose mocked (no pyrosetta)."""
    fr = FastRelax(output_dir=str(tmp_path))

    # Pretend the venv python exists.
    env_python = tmp_path / "env" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.touch()
    monkeypatch.setenv("FASTRELAX_ENV", str(tmp_path / "env"))

    # Stub the in-process RDKit prep to avoid needing real ligands/params.
    prepared = tmp_path / "a.for_pose.pdb"
    prepared.touch()
    monkeypatch.setattr(
        FastRelax, "_prepare_pdb_for_pose",
        lambda self, p, s, c: (prepared, "X01", None),
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        spec = json.loads(Path(cmd[-1]).read_text())
        captured["spec"] = spec
        out = json.dumps(
            {"relaxed_path": spec["out_pdb"], "score": -9.0}
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    relaxed, score = fr._relax_one("/data/a_0_chai.pdb", "CCO", None)

    assert score == -9.0
    assert relaxed.endswith("a_0_chai_relaxed.pdb")
    assert captured["cmd"][0] == str(env_python)
    assert captured["cmd"][1].endswith("_fastrelax_worker.py")
    assert captured["spec"]["ligand_resname"] == "X01"
    assert captured["spec"]["mode"] == "ligand_focused"


def test_relax_one_worker_failure_raises(tmp_path, monkeypatch):
    """Non-zero worker exit -> RuntimeError (execute() then swallows it)."""
    fr = FastRelax(output_dir=str(tmp_path))
    env_python = tmp_path / "env" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.touch()
    monkeypatch.setenv("FASTRELAX_ENV", str(tmp_path / "env"))
    prepared = tmp_path / "a.for_pose.pdb"
    prepared.touch()
    monkeypatch.setattr(
        FastRelax, "_prepare_pdb_for_pose",
        lambda self, p, s, c: (prepared, "X01", None),
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 1, stdout="", stderr="rosetta boom"
        ),
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        fr._relax_one("/data/a_0_chai.pdb", "CCO", None)
```

- [ ] **Step 4: Run to verify all pass**

Run: `pytest tests/test_fastrelax_subprocess.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_fastrelax_subprocess.py
git commit -m "test(fastrelax): unit-test venv resolution, stdout parsing, and _relax_one subprocess seam"
```

---

### Task 5: Align the pyrosetta-gated regression tests with venv gating

**Files:**
- Modify: `tests/test_fastrelax_preserves_ligand_atoms.py:70-72`
- Modify: `tests/test_fastrelax_boltz_atom_names.py:62-64`

**Interfaces:**
- Consumes: `_pyrosetta_available` (now a venv check, Task 2).

**Context:** These two tests import `_pyrosetta_available` and skip on it. After
Task 2 it means "the RosettaFastRelax venv is reachable" instead of "pyrosetta
importable in-env". The tests already call `FastRelax._relax_one` end-to-end, which
now routes through the venv subprocess — so on a machine WITH the venv they
exercise the real subprocess path, and on a machine without it they skip. Only the
skip `reason` string needs updating for accuracy; no logic change.

- [ ] **Step 1: Update the skip reason in `test_fastrelax_preserves_ligand_atoms.py`**

Change the decorator (`tests/test_fastrelax_preserves_ligand_atoms.py:70-73`):

```python
@pytest.mark.skipif(
    not _pyrosetta_available(),
    reason="RosettaFastRelax venv not reachable (set FASTRELAX_ENV)",
)
```

- [ ] **Step 2: Update the skip reason in `test_fastrelax_boltz_atom_names.py`**

Change the decorator (`tests/test_fastrelax_boltz_atom_names.py:62-65`):

```python
@pytest.mark.skipif(
    not _pyrosetta_available(),
    reason="RosettaFastRelax venv not reachable (set FASTRELAX_ENV)",
)
```

- [ ] **Step 3: Run the two files + the full suite**

Run: `pytest tests/test_fastrelax_preserves_ligand_atoms.py tests/test_fastrelax_boltz_atom_names.py -q`
Expected: PASS or SKIP (skipped on login node where the venv may still be
reachable via NFS — that's fine; if reachable they run the real subprocess).

Run: `pytest tests/ tools/test_pipeline.py -q`
Expected: no NEW failures vs the 242-passed/5-skipped baseline (skip count may
shift by ±2 depending on venv reachability; 0 failures is the hard requirement).

- [ ] **Step 4: Commit**

```bash
git add tests/test_fastrelax_preserves_ligand_atoms.py tests/test_fastrelax_boltz_atom_names.py
git commit -m "test(fastrelax): update pyrosetta-gate skip reasons to venv reachability"
```

---

### Task 6: Full-DAG GPU smoke (modular runner) + validation

**Files:**
- Create: `tools/smoke/run_full_dag_smoke.sbatch`

**Context:** The existing `tools/gpu_run/run_allmodules.sbatch` + `run.yml` already
enable all 15 steps (incl. fastrelax + placer). This task adds a smoke wrapper that
(a) runs it against the FRESH handoff env, (b) exports `FASTRELAX_ENV`, and (c) adds
explicit PASS/FAIL validation for fastrelax + placer + the final pickle. Reuse
`tools/gpu_run/run.yml` and `tools/gpu_run/make_input.py` as-is (no new config).

**Interfaces:**
- Consumes: `structurezyme run`, `tools/gpu_run/run.yml`, `tools/gpu_run/make_input.py`.
- Produces: a run dir at `<output_root>/lherrmann/gpu-allmodules-01/` with
  `checkpoints/{fastrelax,placer}.pkl`, `superimposition/fastrelax/*_relaxed.pdb`,
  `placer/*.csv`, and `geometricfiltering/.../structural_features_final.pkl`.

- [ ] **Step 1: Write the smoke sbatch**

Create `tools/smoke/run_full_dag_smoke.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=sz_fulldag_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Full-DAG GPU smoke: all 15 steps incl. fastrelax (subprocess -> RosettaFastRelax
# venv) and placer, against the FRESH handoff env. Submit from repo root:
#   sbatch --output=$PWD/smoke_logs/%x_%j.out --error=$PWD/smoke_logs/%x_%j.err \
#          tools/smoke/run_full_dag_smoke.sbatch
# (Logs MUST be on shared storage; override the sbatch --output/--error above.)
set -euo pipefail

STRUCTUREZYME_ENV="${STRUCTUREZYME_ENV:-/mnt/storage01/home/lherrmann/envs/structurezyme}"
FASTRELAX_ENV="${FASTRELAX_ENV:-/mnt/labs/data/mora/software/RosettaFastRelax/env}"
export FASTRELAX_ENV
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STRUCTUREZYME_ENV"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$REPO_ROOT"

echo "host=$(hostname) python=$(which python) fastrelax_env=$FASTRELAX_ENV"
nvidia-smi | head -20 || true

python tools/gpu_run/make_input.py --output tools/gpu_run/input.pkl
structurezyme run --config tools/gpu_run/run.yml --host default

RUN_DIR="/mnt/labs/data/mora/structurezyme_runs/lherrmann/gpu-allmodules-01"
echo "---- VALIDATION ----"
python - "$RUN_DIR" <<'PY'
import sys, glob, pandas as pd
from pathlib import Path
run = Path(sys.argv[1]); rc = 0

def check(name, ok, detail=""):
    global rc
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok: rc = 1

# fastrelax: checkpoint + relaxed pdbs + score column populated
fr_ckpt = run / "checkpoints" / "fastrelax.pkl"
check("fastrelax checkpoint exists", fr_ckpt.exists(), str(fr_ckpt))
relaxed = list((run / "superimposition" / "fastrelax").rglob("*_relaxed.pdb"))
check("fastrelax produced relaxed pdbs", len(relaxed) > 0, f"{len(relaxed)} files")
if fr_ckpt.exists():
    fr = pd.read_pickle(fr_ckpt)
    has = "fastrelax_score" in fr.columns and any(
        bool(v) and any(v.values()) for v in fr["fastrelax_score"] if isinstance(v, dict)
    )
    check("fastrelax_score populated", has)

# placer: checkpoint + CSV + placer_* columns populated
pl_ckpt = run / "checkpoints" / "placer.pkl"
check("placer checkpoint exists", pl_ckpt.exists(), str(pl_ckpt))
csvs = glob.glob(str(run / "placer" / "*.csv"))
check("placer produced a CSV", len(csvs) > 0, f"{len(csvs)} files")
if pl_ckpt.exists():
    pl = pd.read_pickle(pl_ckpt)
    cols = [c for c in pl.columns if c.startswith("placer_")]
    populated = any(pl[c].notna().any() for c in cols) if cols else False
    check("placer_* columns populated", populated, f"cols={cols}")

# end-to-end: final structural features pickle
finals = list(run.rglob("structural_features_final.pkl"))
check("structural_features_final.pkl exists", len(finals) > 0,
      str(finals[0]) if finals else "MISSING")

print("SMOKE RESULT:", "PASS" if rc == 0 else "FAIL")
sys.exit(rc)
PY
```

- [ ] **Step 2: Shell-lint the sbatch (parse only, do not submit here)**

Run: `bash -n tools/smoke/run_full_dag_smoke.sbatch && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/smoke/run_full_dag_smoke.sbatch
git commit -m "test(smoke): add full-DAG GPU smoke (modular runner) validating fastrelax + placer + final pickle"
```

- [ ] **Step 4: Submit on GPU and confirm PASS (human-in-the-loop)**

Run:
```bash
mkdir -p smoke_logs
sbatch --output="$PWD/smoke_logs/%x_%j.out" --error="$PWD/smoke_logs/%x_%j.err" \
       tools/smoke/run_full_dag_smoke.sbatch
```
Then poll `squeue -u $USER` and, on completion, inspect the log tail for
`SMOKE RESULT: PASS`. Expected: the VALIDATION block prints all `[PASS]` and
`SMOKE RESULT: PASS`. If any `[FAIL]`, debug that step (see docs/known-issues.md)
before proceeding. Determine the concrete `placer_predict_ligand` value here: the
`run.yml` default is `LIG`; if PLACER emits no CSV / empty scores for CalB, inspect
the prepared PDB HETATM records under
`<run>/superimposition/preparedfiles_for_superimposition/*.pdb` and set
`steps.placer.placer_predict_ligand` in `run.yml` accordingly, then re-run with
`force: [placer]`.

---

### Task 7: Documentation

**Files:**
- Modify: `docs/known-issues.md`
- Modify: `docs/getting_started.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Add a FastRelax + PLACER section to `docs/known-issues.md`**

Append a section documenting:
- FastRelax runs pyrosetta **out-of-process** in the RosettaFastRelax venv
  (`/mnt/labs/data/mora/software/RosettaFastRelax/env`), overridable via
  `FASTRELAX_ENV`; pyrosetta is licensed/heavy and deliberately NOT in
  `environment.yml`. Enabling `fastrelax` without the venv raises
  `FileNotFoundError` at first relax.
- PLACER runs its own env (`/mnt/labs/data/mora/software/PLACER/env`, host-profile
  `placer_env_path`) and needs `steps.placer.placer_predict_ligand` set; weights
  `PLACER_model_1.pt` ship with the shared install.
- Both are default-disabled and require the shared-filesystem installs.

Add this text:

```markdown
## FastRelax & PLACER: external, out-of-process steps (default-disabled)

`fastrelax` and `placer` are opt-in steps that shell out to separate shared
installs under `/mnt/labs/data/mora/software/`; neither pyrosetta nor PLACER is
in `environment.yml` (both are licensed/heavy/external).

- **FastRelax** runs PyRosetta in a subprocess against the RosettaFastRelax venv
  at `/mnt/labs/data/mora/software/RosettaFastRelax/env` (py3.12). Override with
  `FASTRELAX_ENV=<env-dir>`. Enable via `steps.fastrelax.enabled: true`. If the
  venv is unreachable, the step raises `FileNotFoundError` at the first pose.
- **PLACER** runs `run_PLACER.py` under `/mnt/labs/data/mora/software/PLACER/env`
  (py3.10; weights `PLACER_model_1.pt` included). Enable via
  `steps.placer.enabled: true` AND set `steps.placer.placer_predict_ligand`
  (chain-resname-resnum, e.g. `A-HEM-154`, or the docked ligand resname `LIG`);
  the step raises if it is unset. Override the env with
  `paths.placer_env_path` or the host profile.
```

- [ ] **Step 2: Add an enablement note to `docs/getting_started.md`**

Add a short subsection near the smoke/GPU instructions:

```markdown
### Optional steps: FastRelax and PLACER

Both are disabled by default and depend on shared installs (not in the conda
env). To run the full 15-step DAG:

1. Ensure the shared installs are reachable:
   - FastRelax: `/mnt/labs/data/mora/software/RosettaFastRelax/env`
     (or set `FASTRELAX_ENV`).
   - PLACER: `/mnt/labs/data/mora/software/PLACER/env` (host-profile
     `placer_env_path`).
2. In your run config, set `steps.fastrelax.enabled: true` and
   `steps.placer.enabled: true` (with `steps.placer.placer_predict_ligand`).
   See `tools/gpu_run/run.yml` for a full-DAG example.
3. Smoke it on GPU: `sbatch tools/smoke/run_full_dag_smoke.sbatch`
   (set `--output/--error` to shared storage).
```

- [ ] **Step 3: Commit**

```bash
git add docs/known-issues.md docs/getting_started.md
git commit -m "docs: document FastRelax (FASTRELAX_ENV subprocess) and PLACER enablement"
```

---

## Task Dependency Summary

- Task 1 (worker) → independent, but Task 3 depends on it.
- Task 2 (helpers) → Task 3, Task 4, Task 5 depend on it.
- Task 3 (`_relax_one` refactor) → depends on 1 + 2.
- Task 4 (unit tests) → depends on 2 + 3.
- Task 5 (gated-test reasons) → depends on 2.
- Task 6 (GPU smoke) → depends on 1-3 landed (needs real env); Step 4 is
  human-in-the-loop GPU submission.
- Task 7 (docs) → independent; can land any time after the design is fixed.

Recommended order: 1 → 2 → 3 → 4 → 5 → 6 → 7.
