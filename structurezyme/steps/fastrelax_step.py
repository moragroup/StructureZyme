"""FastRelax step: rank docked poses per engine and relax the top-K via
Rosetta's FastRelax mover.

Unlike the other per-engine prepare steps, `FastRelax` takes the three
`*_files_for_superimposition` list-columns directly (its input is the
output of `_prepare_files_for_superimposition`), because ranking/selection
is inherently cross-column -- it needs to see Chai, Boltz, and Vina poses
for the same entry together to apply per-engine top-K selection and
rewrite each engine's list in place.

The pure-Python parts (column validation, per-engine top-K ranking, and
dict-key <-> file-path matching) live at module scope and are always
importable. The PyRosetta-dependent parts (`_relax_one`) import
`pyrosetta` lazily inside the method so that `import structurezyme` does
not require PyRosetta to be installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from ..utils.helpers import valid_file_list
from ..utils.ligand_params import (
    LigandParams,
    LigandParamsError,
    canonical_smiles,
    generate_ligand_params,
)
from .step import Step

logger = logging.getLogger(__name__)

# Module-level guard: PyRosetta's `init()` is not safely re-entrant across
# repeated calls in some versions. This flag ensures it runs at most once
# per process, no matter how many `_relax_one` calls we make.
_pyrosetta_initialized = False


def _pyrosetta_available() -> bool:
    """True if `pyrosetta` is importable in the current environment.

    Mirrors `_squidly_cli_available` in `tests/test_squidly_step.py`:
    used to gate smoke tests that require the actual Rosetta binary.
    """
    try:
        import pyrosetta  # noqa: F401
        return True
    except Exception:
        return False


def _confidence_key_from_path(path: str, engine: str):
    """Map a docked-structure file path to its confidence-dict key.

    Chai/Boltz: strip the trailing `_{engine}` suffix from the file's stem
    (e.g. `"Q97WW0_0_chai"` -> `"Q97WW0_0"`, `"Q97WW0_model_0_boltz"` ->
    `"Q97WW0_model_0"`).

    Vina: does NOT use suffix-stripping. `vina_affinities` is keyed by the
    bare 1-indexed pose number, extracted as `int(stem.split('_')[-2])`,
    mirroring `extract_vina_index` in `structurezyme/utils/helpers.py`
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


def _assign_resname(prefix: str, index: int) -> str:
    """Return a 3-letter code like `S01`, `C12`.

    `prefix` is 'S' (substrate) or 'C' (cofactor). `index` is 1-based
    and must fit in 2 digits (max 99 unique ligands per class per run,
    which is plenty for enzyme screening).
    """
    if index < 1 or index > 99:
        raise ValueError(
            f"resname index out of range (1..99): {index}. Too many "
            f"distinct ligands in one run."
        )
    return f"{prefix}{index:02d}"


def _count_ligand_heavy_atoms_by_chain(pdb_text: str) -> dict[str, int]:
    """Map each chain ID to the count of non-H HETATM atoms in it.

    Used to match a PDB's ligand chains against expected SMILES atom
    counts, so we can assign the right resname to each chain.
    """
    counts: dict[str, int] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("HETATM"):
            continue
        element = line[76:78].strip() if len(line) >= 78 else ""
        if not element:
            atom_name = line[12:16].strip()
            element = atom_name.lstrip("0123456789")[:1].upper()
        if element == "H":
            continue
        chain = line[21]
        counts[chain] = counts.get(chain, 0) + 1
    return counts


def _rewrite_ligand_resnames(
    pdb_text: str,
    chain_to_resname: dict[str, str],
) -> str:
    """Return `pdb_text` with HETATM residue names rewritten by chain.

    For every HETATM line whose chain ID is a key of
    `chain_to_resname`, replace the 3-letter residue name at columns
    18-20 with the mapped code. Non-HETATM lines pass through
    unchanged.
    """
    out_lines = []
    for line in pdb_text.splitlines(keepends=False):
        if line.startswith("HETATM") and line[21] in chain_to_resname:
            new_resname = chain_to_resname[line[21]]
            line = line[:17] + new_resname + line[20:]
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


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
    (see `_apply_relaxed_paths`). Also attaches a `fastrelax_score`
    dict column recording each relaxed pose's final Rosetta score,
    keyed the same way as the engine's confidence dict.

    Per-pose failures (Rosetta exception, missing/corrupt PDB, ...) are
    logged and swallowed by `execute`; the pose's original path is left
    in place and the run continues, matching the existing
    per-entry-tolerant pattern used elsewhere in the pipeline.
    """

    def __init__(
        self,
        output_dir: str,
        *,
        substrate_smiles_col: str = "substrate_smiles",
        cofactor_smiles_col: str = "cofactor_smiles",
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
        self.substrate_smiles_col = substrate_smiles_col
        self.cofactor_smiles_col = cofactor_smiles_col
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

        # Populated at execute() start; keyed by canonical SMILES.
        # e.g. {"C1=CC=C2C(=C1)C=CN2": "S01", ...}
        self._substrate_resnames: dict[str, str] = {}
        self._cofactor_resnames: dict[str, str] = {}
        # LigandParams objects by canonical SMILES, for -extra_res_fa init.
        self._ligand_params: dict[str, LigandParams] = {}

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

    def _register_ligands(self, df: pd.DataFrame) -> None:
        """Walk `df`, assign a unique 3-letter code per unique SMILES,
        and generate a `.params` file for each.

        Populates `self._substrate_resnames`, `self._cofactor_resnames`,
        and `self._ligand_params`. Missing substrate SMILES on any row
        is a hard error (FastRelax cannot run without a substrate).
        Missing/empty cofactor SMILES is allowed and simply means that
        row has no cofactor to relax.

        Called once at the start of `execute()`; safe to call again on
        the same instance (idempotent -- unique SMILES already
        registered are reused).

        Params files land in `self.output_dir / 'params'`.
        """
        params_dir = self.output_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)

        substrate_col = self.substrate_smiles_col
        cofactor_col = self.cofactor_smiles_col

        if substrate_col not in df.columns:
            raise ValueError(
                f"FastRelax requires a substrate SMILES column "
                f"{substrate_col!r}; DataFrame columns are {list(df.columns)}"
            )

        # Substrate: hard-required per row.
        for idx, smi in enumerate(df[substrate_col].tolist()):
            if smi is None or (isinstance(smi, str) and not smi.strip()):
                raise ValueError(
                    f"FastRelax row {idx} has empty/None substrate SMILES "
                    f"in column {substrate_col!r}; substrate is required"
                )
            canon = canonical_smiles(smi)
            if canon in self._substrate_resnames:
                continue
            # 'X' prefix avoids collision with Rosetta's built-in
            # non-canonical amino acid codes (A/B/C/G/M/S/U/V families).
            resname = _assign_resname("X", len(self._substrate_resnames) + 1)
            self._substrate_resnames[canon] = resname
            self._ligand_params[canon] = generate_ligand_params(
                smiles=smi, resname=resname, cache_dir=params_dir,
            )
            logger.info(
                "FastRelax registered substrate %s -> %s", canon, resname,
            )

        # Cofactor: optional. Missing column or empty entries are fine.
        if cofactor_col in df.columns:
            for smi in df[cofactor_col].tolist():
                if smi is None or (isinstance(smi, str) and not smi.strip()):
                    continue
                canon = canonical_smiles(smi)
                if canon in self._cofactor_resnames:
                    continue
                # 'Z' prefix avoids collision with Rosetta's built-in
                # non-canonical amino acid codes (see substrate note above).
                resname = _assign_resname("Z", len(self._cofactor_resnames) + 1)
                self._cofactor_resnames[canon] = resname
                self._ligand_params[canon] = generate_ligand_params(
                    smiles=smi, resname=resname, cache_dir=params_dir,
                )
                logger.info(
                    "FastRelax registered cofactor %s -> %s", canon, resname,
                )

    def _prepare_pdb_for_pose(
        self,
        pdb_path: str | Path,
        substrate_smiles: str | None,
        cofactor_smiles: str | None,
    ) -> tuple[Path, str | None, str | None]:
        """Rewrite the input PDB's ligand resnames to the codes we
        assigned in `_register_ligands`, and drop the result next to
        the original with a `.for_pose.pdb` suffix.

        Returns `(prepared_pdb_path, substrate_resname, cofactor_resname)`.
        A resname is `None` if the corresponding SMILES was
        `None`/empty or if no matching HETATM chain was found.

        Chain-to-ligand assignment: for each HETATM chain in the PDB,
        count non-H atoms and match against the expected heavy-atom
        count of substrate and cofactor. Ties are unusual (substrate
        and cofactor with identical heavy-atom counts) but if they
        happen we skip the ambiguous rename and log a warning.

        Raises `RuntimeError` if the input PDB has HETATM chains that
        don't match either registered ligand -- silently letting them
        through would recreate the original bug (they'd fall back to
        pdb_LIG's 29-atom template).
        """
        pdb_path = Path(pdb_path)
        pdb_text = pdb_path.read_text()
        chain_counts = _count_ligand_heavy_atoms_by_chain(pdb_text)

        sub_canon = canonical_smiles(substrate_smiles) if substrate_smiles else None
        cof_canon = canonical_smiles(cofactor_smiles) if cofactor_smiles else None

        sub_resname = self._substrate_resnames.get(sub_canon) if sub_canon else None
        cof_resname = self._cofactor_resnames.get(cof_canon) if cof_canon else None

        sub_expected = (
            self._ligand_params[sub_canon].params_path if sub_canon else None
        )
        cof_expected = (
            self._ligand_params[cof_canon].params_path if cof_canon else None
        )

        def _params_heavy(params_path: Path | None) -> int | None:
            if params_path is None:
                return None
            n = 0
            for line in params_path.read_text().splitlines():
                if not line.startswith("ATOM"):
                    continue
                parts = line.split()
                if len(parts) >= 2 and not parts[1].startswith("H"):
                    n += 1
            return n

        sub_target = _params_heavy(sub_expected)
        cof_target = _params_heavy(cof_expected)

        chain_to_resname: dict[str, str] = {}
        for chain, count in chain_counts.items():
            hits = []
            if sub_target is not None and count == sub_target and sub_resname:
                hits.append(("substrate", sub_resname))
            if cof_target is not None and count == cof_target and cof_resname:
                hits.append(("cofactor", cof_resname))
            if len(hits) == 1:
                chain_to_resname[chain] = hits[0][1]
            elif len(hits) > 1:
                logger.warning(
                    "FastRelax: chain %s has %d heavy atoms matching both "
                    "substrate (%s) and cofactor (%s); skipping rename",
                    chain, count, sub_resname, cof_resname,
                )
            else:
                raise RuntimeError(
                    f"FastRelax: {pdb_path} chain {chain} has {count} heavy "
                    f"atoms, matching neither substrate ({sub_target}) nor "
                    f"cofactor ({cof_target}). Cannot safely assign a "
                    f"3-letter code -- Rosetta would fall back to the "
                    f"generic 29-atom LIG template and corrupt the ligand."
                )

        new_text = _rewrite_ligand_resnames(pdb_text, chain_to_resname)
        prepared = pdb_path.with_suffix(".for_pose.pdb")
        prepared.write_text(new_text)
        return prepared, sub_resname, cof_resname

    def _relax_one(
        self,
        pdb_path,
        substrate_smiles: str | None = None,
        cofactor_smiles: str | None = None,
    ) -> tuple[str, float]:
        """Relax one PDB with PyRosetta's FastRelax mover.

        Returns `(relaxed_pdb_path, final_score)`. Raises on error --
        `execute()` wraps calls in try/except so per-pose failures don't
        abort the run.

        `substrate_smiles` / `cofactor_smiles` are the row's ligand
        SMILES; they select which `.params` file was assigned to each
        chain and drive the LIG-to-code rename applied before
        `pose_from_pdb`. Must be set unless the pose really has no
        non-canonical ligand (e.g. protein-only relaxation) -- in
        which case skip this method entirely.

        `pyrosetta` is imported lazily inside this method so that
        `import structurezyme` doesn't require PyRosetta.
        """
        try:
            import pyrosetta
        except ImportError as e:
            raise RuntimeError(
                "PyRosetta is required for FastRelax; install it into the "
                "current env (see /mnt/labs/data/mora/software/"
                "RosettaFastRelax/ for the shared installer)."
            ) from e

        global _pyrosetta_initialized
        try:
            if not _pyrosetta_initialized:
                # Pass every registered params file to Rosetta via
                # -extra_res_fa. Without this Rosetta silently maps any
                # unknown LIG residue to a 29-atom generic template and
                # corrupts the ligand chemistry. See docs/known-issues.md.
                extra_res_fa = " ".join(
                    str(lp.params_path) for lp in self._ligand_params.values()
                )
                init_opts = "-mute all"
                if extra_res_fa:
                    init_opts = f"{init_opts} -extra_res_fa {extra_res_fa}"
                pyrosetta.init(extra_options=init_opts, silent=True)
                _pyrosetta_initialized = True

            prepared_pdb, sub_resname, cof_resname = self._prepare_pdb_for_pose(
                pdb_path, substrate_smiles, cofactor_smiles,
            )
            pose = pyrosetta.pose_from_pdb(str(prepared_pdb))
            # Ligand resname used by ligand_focused mode's neighborhood
            # selector. Prefer the substrate; fall back to the cofactor.
            ligand_resname_for_selector = sub_resname or cof_resname

            movemap = None
            movemap_factory = None
            if self.mode == "ligand_focused":
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

                if ligand_resname_for_selector is None:
                    raise RuntimeError(
                        f"FastRelax mode='ligand_focused' requires either "
                        f"a substrate or cofactor SMILES on the row; both "
                        f"were empty for {pdb_path}."
                    )
                ligand_sel = ResidueNameSelector()
                ligand_sel.set_residue_name3(ligand_resname_for_selector)
                shell_sel = NeighborhoodResidueSelector(
                    ligand_sel, self.shell_radius, True
                )

                movemap_factory = MoveMapFactory()
                movemap_factory.all_bb(False)
                movemap_factory.all_chi(False)
                movemap_factory.add_bb_action(move_map_action.mm_enable, shell_sel)
                movemap_factory.add_chi_action(move_map_action.mm_enable, shell_sel)

                coord_gen = CoordinateConstraintGenerator()
                coord_gen.set_residue_selector(shell_sel)
                coord_gen.set_sd(1.0 / max(self.constraint_weight, 1e-6))
                add_csts = AddConstraints()
                add_csts.add_generator(coord_gen)
                add_csts.apply(pose)
            elif self.mode == "full":
                movemap = pyrosetta.MoveMap()
                movemap.set_bb(True)
                movemap.set_chi(True)
            else:
                raise ValueError(f"Unknown FastRelax mode: {self.mode!r}")

            scorefxn = pyrosetta.create_score_function(self.scorefunction)
            if self.mode == "ligand_focused":
                from pyrosetta.rosetta.core.scoring import ScoreType
                scorefxn.set_weight(
                    ScoreType.coordinate_constraint, self.constraint_weight
                )

            relax = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn)
            if movemap_factory is not None:
                relax.set_movemap_factory(movemap_factory)
            elif movemap is not None:
                relax.set_movemap(movemap)
            relax.apply(pose)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            relaxed_path = self.output_dir / f"{Path(pdb_path).stem}_relaxed.pdb"
            pose.dump_pdb(str(relaxed_path))

            return (str(relaxed_path), float(scorefxn(pose)))
        except Exception as e:
            logger.error("FastRelax._relax_one failed for %s: %s", pdb_path, e)
            raise

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)
        # Assign a unique 3-letter code per unique SMILES and generate
        # a .params file for each. Must run before PyRosetta init so
        # -extra_res_fa can reference every params file up-front.
        self._register_ligands(df)

        # Per-row selection: build a dict[engine, list[selected_paths]]
        # for each row. This step is pyrosetta-independent and lets us
        # validate + shape the work before touching Rosetta.
        per_row_selected: list[dict] = [
            self._rank_and_select(row) for _, row in df.iterrows()
        ]

        engine_specs = (
            ("chai", self.chai_files_col),
            ("boltz", self.boltz_files_col),
            ("vina", self.vina_files_col),
        )

        df = df.copy()
        # Make the files-columns object-typed so we can assign list values
        # cleanly, and allocate the fastrelax_score column.
        for _, files_col in engine_specs:
            df[files_col] = df[files_col].astype(object)
        df["fastrelax_score"] = [{} for _ in range(len(df))]

        for row_idx, (df_idx, row) in enumerate(df.iterrows()):
            selected = per_row_selected[row_idx]
            fastrelax_score: dict = {"chai": {}, "boltz": {}, "vina": {}}
            row_sub_smiles = row.get(self.substrate_smiles_col)
            row_cof_smiles = row.get(self.cofactor_smiles_col)

            for engine, files_col in engine_specs:
                selected_paths = selected.get(engine, [])
                relaxed_map: dict = {}
                for path in selected_paths:
                    try:
                        relaxed_path, score = self._relax_one(
                            path,
                            substrate_smiles=row_sub_smiles,
                            cofactor_smiles=row_cof_smiles,
                        )
                    except Exception as e:
                        logger.error(
                            "FastRelax: skipping pose %s for engine %s: %s",
                            path,
                            engine,
                            e,
                        )
                        # Leave the original path in place -- see spec
                        # Behavior step 6. No entry in `relaxed_map` for
                        # this path, so `_apply_relaxed_paths` falls back
                        # to the original.
                        continue
                    relaxed_map[path] = relaxed_path
                    key = _confidence_key_from_path(path, engine)
                    fastrelax_score[engine][key] = score

                original_files = row[files_col]
                if not valid_file_list(original_files):
                    # Nothing to rewrite for this engine on this row.
                    continue
                new_files = _apply_relaxed_paths(
                    list(original_files),
                    selected_paths,
                    relaxed_map,
                    self.drop_unrelaxed,
                )
                df.at[df_idx, files_col] = new_files

            df.at[df_idx, "fastrelax_score"] = fastrelax_score

        return df

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
