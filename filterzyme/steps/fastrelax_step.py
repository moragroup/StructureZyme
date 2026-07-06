"""FastRelax step: pure ranking/selection logic (no PyRosetta dependency).

Unlike the other per-engine prepare steps, `FastRelax` takes the three
`*_files_for_superimposition` list-columns directly (its input is the
output of `_prepare_files_for_superimposition`), because ranking/selection
is inherently cross-column -- it needs to see Chai, Boltz, and Vina poses
for the same entry together to apply per-engine top-K selection and
rewrite each engine's list in place.

This module covers everything in the FastRelax step that has no PyRosetta
dependency: column validation, per-engine top-K ranking (including the
ascending/descending direction switch), and dict-key <-> file-path
matching. The actual PyRosetta-dependent relax call (`_relax_one`,
`execute`) is implemented in a later task.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from ..utils.helpers import valid_file_list
from .step import Step


def _confidence_key_from_path(path: str, engine: str):
    """Map a docked-structure file path to its confidence-dict key.

    Chai/Boltz: strip the trailing `_{engine}` suffix from the file's stem
    (e.g. `"Q97WW0_0_chai"` -> `"Q97WW0_0"`, `"Q97WW0_model_0_boltz"` ->
    `"Q97WW0_model_0"`).

    Vina: does NOT use suffix-stripping. `vina_affinities` is keyed by the
    bare 1-indexed pose number, extracted as `int(stem.split('_')[-2])`,
    mirroring `extract_vina_index` in `filterzyme/utils/helpers.py`
    (`add_metrics`).
    """
    stem = Path(path).stem
    if engine == "vina":
        return int(stem.split("_")[-2])
    suffix = f"_{engine}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _select_top_k(paths: list[str], confidence: dict, engine: str, top_k: int) -> list[str]:
    """Rank `paths` by their `confidence` dict value and keep the top `top_k`.

    Sort direction depends on the engine: descending for chai/boltz
    (higher confidence score is better), ascending for vina (more negative
    affinity is better). Paths whose confidence key is absent from
    `confidence` are excluded entirely -- treated as unrankable, not
    artificially ranked last.
    """
    scored = []
    for p in paths:
        key = _confidence_key_from_path(p, engine)
        if key in confidence:
            scored.append((p, confidence[key]))
    reverse = engine != "vina"  # descending for chai/boltz, ascending for vina
    scored.sort(key=lambda t: t[1], reverse=reverse)
    return [p for p, _ in scored[:top_k]]


def _apply_relaxed_paths(
    files: list[str],
    selected: list[str],
    relaxed_map: dict,
    drop_unrelaxed: bool,
) -> list[str]:
    """Rewrite an engine's file-path list after relaxation.

    For each original path in `files`: if it was `selected` for relaxation,
    replace it with its relaxed path from `relaxed_map` (falling back to
    the original path if relaxation hasn't produced a mapped entry for it
    yet). If it was not selected, keep it only when `drop_unrelaxed` is
    False.
    """
    selected_set = set(selected)
    out = []
    for f in files:
        if f in selected_set:
            out.append(relaxed_map.get(f, f))
        elif not drop_unrelaxed:
            out.append(f)
    return out


class FastRelax(Step):
    """Rosetta FastRelax step: relax the top-K docked poses per engine.

    This step reads the three `*_files_for_superimposition` list-columns
    produced by `_prepare_files_for_superimposition`, ranks each engine's
    poses by its own confidence-dict column and sort direction, relaxes
    the top `top_k` poses per engine with PyRosetta's FastRelax mover, and
    rewrites each engine's file-path list in place with the relaxed paths
    (see `_apply_relaxed_paths`).

    The PyRosetta-dependent relax call (`_relax_one`, `execute`) is
    implemented in a later task; this constructor and `_validate_input`/
    `_rank_and_select` contain only pure Python logic.
    """

    def __init__(
        self,
        output_dir: str,
        ligand_resname: str,
        entry_col: str = "Entry",
        chai_files_col: str = "chai_files_for_superimposition",
        boltz_files_col: str = "boltz_files_for_superimposition",
        vina_files_col: str = "vina_files_for_superimposition",
        chai_confidence_col: str = "chai_ptm",
        boltz_confidence_col: str = "boltz2_confidence_score",
        vina_confidence_col: str = "vina_affinities",
        top_k: int = 2,
        drop_unrelaxed: bool = True,
        mode: Literal["ligand_focused", "full"] = "ligand_focused",
        shell_radius: float = 8.0,
        constraint_weight: float = 1.0,
        scorefunction: str = "ref2015",
        num_threads: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.ligand_resname = ligand_resname
        self.entry_col = entry_col
        self.chai_files_col = chai_files_col
        self.boltz_files_col = boltz_files_col
        self.vina_files_col = vina_files_col
        self.chai_confidence_col = chai_confidence_col
        self.boltz_confidence_col = boltz_confidence_col
        self.vina_confidence_col = vina_confidence_col
        self.top_k = top_k
        self.drop_unrelaxed = drop_unrelaxed
        self.mode = mode
        self.shell_radius = shell_radius
        self.constraint_weight = constraint_weight
        self.scorefunction = scorefunction
        self.num_threads = num_threads or 1

    def _validate_input(self, df: pd.DataFrame) -> None:
        required = [
            self.entry_col,
            self.chai_files_col,
            self.boltz_files_col,
            self.vina_files_col,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"FastRelax input DataFrame is missing required columns: {missing}"
            )

    def _rank_and_select(self, row: pd.Series) -> dict:
        """Rank and select the top-K poses per engine for a single row.

        Returns `{"chai": [...], "boltz": [...], "vina": [...]}`, each a
        list of the top-`top_k` file paths for that engine. An engine
        whose files-column is empty/NaN for this row contributes an empty
        list (using `valid_file_list` to guard, mirroring
        `Superimposition._superimposition`'s pattern).
        """
        engine_specs = (
            ("chai", self.chai_files_col, self.chai_confidence_col),
            ("boltz", self.boltz_files_col, self.boltz_confidence_col),
            ("vina", self.vina_files_col, self.vina_confidence_col),
        )
        result = {}
        for engine, files_col, confidence_col in engine_specs:
            files = row.get(files_col)
            if not valid_file_list(files):
                result[engine] = []
                continue
            confidence = row.get(confidence_col)
            if not isinstance(confidence, dict):
                confidence = {}
            result[engine] = _select_top_k(files, confidence, engine, self.top_k)
        return result
