"""New consolidated PLACER step.

Replaces the three deleted files (`PLACER_step.py`, `PLACER_forChai_step.py`,
`PLACER_forVina_step.py`), which all declared a colliding `class PLACER(Step)`
and none of which was imported by `pipeline_v2.py`.

This module currently provides:

- `_count_ligands`: module-level helper (ported from the old
  `PLACER_forChai_step._count_ligands` method) that counts distinct ligand
  instances (unique chain + resseq) with a given residue name in a PDB
  file. Promoted from a method to a module-level function so it can be
  unit-tested standalone.

- `class PLACER(Step)`: the step class scaffold. Only `__init__` is
  implemented here — it stores its parameters and validates that
  `placer_script_path` points at an existing file. The actual PLACER
  subprocess invocation and `execute` method are implemented in a later
  task; the class inherits `Step.execute`'s identity pass-through for now
  (same pattern as `FastRelax` at commit 00f530f).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from structurezyme.steps.step import Step

logger = logging.getLogger(__name__)


def _method_count(best_method) -> int:
    """Count comma-separated methods in `best_method`. NaN/empty -> 0."""
    if pd.isna(best_method) or best_method == "":
        return 0
    return len(str(best_method).split(","))


def _has_fused_rank(best_method) -> bool:
    """True if 'fused_rank' is one of the comma-separated tokens."""
    if pd.isna(best_method) or best_method == "":
        return False
    return "fused_rank" in str(best_method).split(",")


def _select_one_per_entry(df: pd.DataFrame, entry_col: str = "Entry") -> pd.DataFrame:
    """Reduce a per-(entry, docked_structure) DataFrame to exactly one row per entry.

    Tie-break order:
      1. Prefer rows where ``is_best == True`` (fall back to full group if none).
      2. Among those, prefer highest ``_method_count(best_method)``.
      3. Final tie-break: alphabetically smallest ``docked_structure``.

    Returns a fresh DataFrame with the original columns (no ``_method_count`` leak)
    and a reset index. Row order matches the first appearance of each entry in
    the input (via ``groupby(..., sort=False)``).
    """
    rows = []
    for _entry, group in df.groupby(entry_col, sort=False):
        pool = group[group["is_best"] == True]  # noqa: E712 (explicit bool compare intentional; NaN-safe)
        if pool.empty:
            pool = group
        pool = pool.copy()
        # fused_rank is authoritative: if any pose in the pool was chosen by
        # the energy-aware fusion, restrict to those before other tie-breaks.
        fused = pool[pool["best_method"].apply(_has_fused_rank)]
        if not fused.empty:
            pool = fused
        pool["_method_count"] = pool["best_method"].apply(_method_count)
        max_count = pool["_method_count"].max()
        pool = pool[pool["_method_count"] == max_count]
        pool = pool.sort_values("docked_structure")
        rows.append(pool.iloc[0])
    result = pd.DataFrame(rows).drop(columns=["_method_count"])
    return result.reset_index(drop=True)


_PLACER_SCORE_COLS = (
    "fape", "lddt", "rmsd", "kabsch", "prmsd", "plddt", "plddt_pde",
)


def _placer_available() -> bool:
    """True iff the shared PLACER install is reachable AND smoke tests opted in.

    Smoke tests require:
      1. Env var ``PLACER_SMOKE=1`` (opt-in — smoke test uses GPU-time; keep default suite fast)
      2. ``/mnt/labs/data/mora/software/PLACER/run_PLACER.py`` exists
      3. ``/mnt/labs/data/mora/software/PLACER/env/bin/python`` exists
    """
    import os
    if os.environ.get("PLACER_SMOKE") != "1":
        return False
    script = Path("/mnt/labs/data/mora/software/PLACER/run_PLACER.py")
    python = Path("/mnt/labs/data/mora/software/PLACER/env/bin/python")
    return script.exists() and python.exists()


def _parse_placer_csv(csv_path: Path | str) -> dict[str, float | None]:
    """Parse a PLACER output CSV and return the top-row scores as a dict.

    PLACER writes rows pre-sorted by its ``--rerank`` metric, so ``iloc[0]`` is
    the best sample. Returns 7 ``placer_*`` keys, all ``None`` if the file is
    missing or malformed. Never raises.
    """
    empty = {f"placer_{c}": None for c in _PLACER_SCORE_COLS}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return empty
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return empty
        row = df.iloc[0]
        result: dict[str, float | None] = {}
        for col in _PLACER_SCORE_COLS:
            if col in df.columns:
                val = row[col]
                result[f"placer_{col}"] = float(val) if pd.notna(val) else None
            else:
                result[f"placer_{col}"] = None
        return result
    except Exception as e:
        logger.warning("Failed to parse PLACER CSV %s: %s", csv_path, e)
        return empty


def _count_ligands(pdb_path: Path | str, ligand_resname: str) -> int:
    """Count distinct ligand instances (unique chain + resseq) in a PDB file.

    Ported from the deleted PLACER_forChai_step.py:55-70.
    Returns 0 on any read/parse error.
    """
    unique_ligands: set[tuple[str, str]] = set()
    target_resname = ligand_resname.strip().upper()
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith(("HETATM", "ATOM")):
                    res_name = line[17:20].strip()
                    if res_name == target_resname:
                        unique_id = (line[21], line[22:26])
                        unique_ligands.add(unique_id)
        return len(unique_ligands)
    except Exception:
        return 0


class PLACER(Step):
    """PLACER step: predict ligand binding poses via the PLACER binary.

    Scaffold only — the `execute` method (which shells out to
    `run_PLACER.py` per entry, then counts predicted ligand instances in
    each output PDB via `_count_ligands`) is implemented in a later task.
    For now, the class inherits `Step.execute`'s identity pass-through,
    matching the `FastRelax` pattern at commit 00f530f.
    """

    def __init__(
        self,
        preparedfiles_dir: Path | str,
        output_dir: Path | str,
        predict_ligand: str,
        entry_col: str = "Entry",
        structure_col: str = "docked_structure",
        placer_script_path: str = "/mnt/labs/data/mora/software/PLACER/run_PLACER.py",
        placer_env_path: str = "/mnt/labs/data/mora/software/PLACER/env",
        nsamples: int = 50,
        rerank: str = "prmsd",
        num_threads: int = 1,
    ):
        self.preparedfiles_dir = Path(preparedfiles_dir)
        self.output_dir = Path(output_dir)
        self.predict_ligand = predict_ligand
        self.entry_col = entry_col
        self.structure_col = structure_col
        self.nsamples = nsamples
        self.rerank = rerank
        self.num_threads = num_threads

        script = Path(placer_script_path)
        if not script.exists():
            raise FileNotFoundError(
                f"PLACER script not found at {placer_script_path}. "
                "Set placer_script_path explicitly or install PLACER."
            )
        self.placer_script_path = script

        env_python = Path(placer_env_path) / "bin" / "python"
        if not env_python.exists():
            raise FileNotFoundError(
                f"PLACER env python not found at {env_python}. "
                "Set placer_env_path explicitly or install PLACER."
            )
        self.placer_env_python = env_python

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_pdb_path(self, row: pd.Series) -> Path:
        """Return the .pdb path in preparedfiles_dir corresponding to a DataFrame row.

        Mirrors GeneralGeometricFiltering's convention
        (geometric_filtering_cofactor_MCS.py:466).
        """
        return self.preparedfiles_dir / f"{row[self.structure_col]}.pdb"

    def _build_cmd(self, pdb_path: Path, n_ligands: int) -> list[str]:
        """Build the full argv for a single PLACER invocation.

        Multi-ligand adds ``--predict_multi``; single-ligand does not (spec 386-389).
        """
        cmd = [
            str(self.placer_env_python),
            str(self.placer_script_path),
            "--ifile", str(pdb_path),
            "--odir", str(self.output_dir),
            "--rerank", self.rerank,
            "-n", str(self.nsamples),
            "--predict_ligand", self.predict_ligand,
        ]
        if n_ligands > 1:
            cmd.append("--predict_multi")
        return cmd

    def _validate_input(self, df: pd.DataFrame) -> None:
        # Entry column name is configurable; check the configured name plus the rest
        required = {self.entry_col, self.structure_col, "is_best", "best_method"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"PLACER.execute: input DataFrame missing required columns: {sorted(missing)}"
            )

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run PLACER once per entry and merge the top-row scores back.

        Per-entry failures are logged and produce ``None`` for all 8 new columns;
        never raises out of ``execute()`` (spec Behavior step 8).
        """
        import subprocess

        self._validate_input(df)
        reduced = _select_one_per_entry(df, entry_col=self.entry_col)

        new_cols: dict[str, list] = {f"placer_{c}": [] for c in _PLACER_SCORE_COLS}
        new_cols["placer_dir"] = []

        for _idx, row in reduced.iterrows():
            entry = row[self.entry_col]
            pdb_path = self._build_pdb_path(row)
            empty = {f"placer_{c}": None for c in _PLACER_SCORE_COLS}
            empty["placer_dir"] = None

            if not pdb_path.exists():
                # Missing prepared PDB = upstream pipeline failure. Fail loudly
                # rather than emit an empty (None) score row that looks OK.
                raise FileNotFoundError(
                    f"PLACER: prepared PDB not found for entry {entry}: {pdb_path}"
                )

            n_ligands = _count_ligands(pdb_path, self.predict_ligand)
            cmd = self._build_cmd(pdb_path, n_ligands)

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                # Retry as single-ligand if multi-ligand crashed with AssertionError
                # (mirrors old PLACER_forChai_step.py:150-158 fallback)
                if (
                    result.returncode != 0
                    and "AssertionError" in result.stderr
                    and n_ligands > 1
                ):
                    logger.info(
                        "PLACER: multi-ligand failed for %s; retrying single-ligand.",
                        entry,
                    )
                    cmd = self._build_cmd(pdb_path, n_ligands=1)
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=False
                    )
                if result.returncode != 0:
                    logger.error(
                        "PLACER: subprocess failed for entry %s (rc=%d): %s",
                        entry, result.returncode, result.stderr[-500:],
                    )
                    for k, v in empty.items():
                        new_cols[k].append(v)
                    continue

                # Find output CSV. PLACER names it <label>.csv where <label> is
                # derived from input stem + optional --suffix. We didn't pass
                # --suffix, so glob for <stem>*.csv.
                csvs = list(Path(self.output_dir).glob(f"{pdb_path.stem}*.csv"))
                if not csvs:
                    logger.error(
                        "PLACER: no output CSV for entry %s at %s",
                        entry, self.output_dir,
                    )
                    for k, v in empty.items():
                        new_cols[k].append(v)
                    continue

                scores = _parse_placer_csv(csvs[0])
                for k in [f"placer_{c}" for c in _PLACER_SCORE_COLS]:
                    new_cols[k].append(scores[k])
                new_cols["placer_dir"].append(str(self.output_dir))
            except Exception as e:
                logger.exception(
                    "PLACER: unexpected error for entry %s: %s", entry, e
                )
                for k, v in empty.items():
                    new_cols[k].append(v)

        # Merge into reduced DataFrame
        out = reduced.reset_index(drop=True).copy()
        for k, v in new_cols.items():
            out[k] = v
        return out
