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

from filterzyme.steps.step import Step

logger = logging.getLogger(__name__)


def _method_count(best_method) -> int:
    """Count comma-separated methods in `best_method`. NaN/empty -> 0."""
    if pd.isna(best_method) or best_method == "":
        return 0
    return len(str(best_method).split(","))


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
        pool["_method_count"] = pool["best_method"].apply(_method_count)
        max_count = pool["_method_count"].max()
        pool = pool[pool["_method_count"] == max_count]
        pool = pool.sort_values("docked_structure")
        rows.append(pool.iloc[0])
    result = pd.DataFrame(rows).drop(columns=["_method_count"])
    return result.reset_index(drop=True)


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
        placer_script_path: str = "/mnt/storage01/home/lherrmann/PLACER_tmp_clone/run_PLACER.py",
        placer_conda_env: str = "placer_env",
        nsamples: int = 50,
        rerank: str = "prmsd",
        num_threads: int = 1,
    ):
        self.preparedfiles_dir = Path(preparedfiles_dir)
        self.output_dir = Path(output_dir)
        self.predict_ligand = predict_ligand
        self.entry_col = entry_col
        self.structure_col = structure_col
        self.placer_conda_env = placer_conda_env
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

        self.output_dir.mkdir(parents=True, exist_ok=True)
