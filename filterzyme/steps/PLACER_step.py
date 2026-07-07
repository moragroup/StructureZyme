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

import pandas as pd  # noqa: F401  (kept for future execute() DataFrame usage)

from filterzyme.steps.step import Step

logger = logging.getLogger(__name__)


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
