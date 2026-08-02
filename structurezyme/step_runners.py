# structurezyme/step_runners.py
"""Modular step runner callables for the resumable pipeline engine.

Each ``run_<step>(ctx, spec) -> pd.DataFrame`` faithfully reproduces the
operations of the corresponding private method on the legacy phase classes in
``structurezyme.pipeline`` (``Docking``, ``Superimposition``,
``GeometricFilters``).  The functions:

  1. Read their input frame(s) via ``ctx.input_frames(spec)`` (the first step,
     ``squidly``, falls back to the seed input DataFrame at
     ``ctx.checkpoint_path("_input")`` written by the adapter/CLI).
  2. Run the same step operations against the *same* on-disk sub-directory
     layout the legacy pipeline used, so the fixed-name intermediate pickles
     (``dockingmetrics.pkl``, ``ligandRMSD.pkl`` ...) remain the inter-step
     contract that the downstream ``enzymetk`` steps rely on.
  3. Return the resulting DataFrame; the Runner pickles it to
     ``checkpoints/<step>.pkl``.

Heavy third-party imports (``enzymetk`` and the concrete step classes) are done
lazily *inside* each function so this module imports cleanly without the heavy
scientific dependencies present, which keeps the registry importable and lets
steps be included/excluded freely.

Directory layout (rooted at the run directory ``ctx.layout.root``), matching the
legacy ``Pipeline.run()``:
  <run>/docking/          docking steps (squidly, chai, boltz, vina, metrics)
  <run>/superimposition/  prepare_files, fastrelax, superimpose, rmsd
  <run>/geometricfiltering/ geometric_filter, fpocket, ligand_sasa, plip
  <run>/placer/           placer
"""
from pathlib import Path

import pandas as pd

from structurezyme.utils.helpers import iter_substrates


# --------------------------------------------------------------------------
# Directory / input helpers
# --------------------------------------------------------------------------
def _docking_dir(ctx) -> Path:
    d = ctx.layout.root / "docking"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _superimp_dir(ctx) -> Path:
    d = ctx.layout.root / "superimposition"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _publish_relaxed_to_prepared(fastrelax_dir: Path, prepared_dir: Path) -> int:
    """Copy fastrelax's ``*_relaxed.pdb`` outputs into the prepared-files dir.

    When fastrelax is enabled it rewrites ``docked_structure`` to the relaxed
    pose names (``<stem>_relaxed``) but writes the relaxed pdbs only under
    ``fastrelax_dir``. The downstream analysis steps (geometric_filter, fpocket,
    ligand_sasa, plip, placer) resolve poses as
    ``prepared_dir / f"{docked_structure}.pdb"``. Copying the relaxed pdbs into
    ``prepared_dir`` under their ``_relaxed`` names makes those lookups resolve
    to the refined structures without touching the individual step classes.

    Returns the number of files copied. Missing dirs / no relaxed files are a
    no-op (returns 0), so the fastrelax-disabled path is unaffected.
    """
    import shutil

    fastrelax_dir = Path(fastrelax_dir)
    prepared_dir = Path(prepared_dir)
    if not fastrelax_dir.is_dir():
        return 0
    prepared_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in fastrelax_dir.glob("*_relaxed.pdb"):
        shutil.copy2(src, prepared_dir / src.name)
        copied += 1
    return copied


def _geo_dir(ctx) -> Path:
    d = ctx.layout.root / "geometricfiltering"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seed_input(ctx) -> pd.DataFrame:
    """The initial input DataFrame, seeded to checkpoints/_input.pkl."""
    p = ctx.checkpoint_path("_input")
    return pd.read_pickle(p)


def _first_input(ctx, spec) -> pd.DataFrame:
    """The single upstream input frame for a step, or the seed frame."""
    frames = ctx.input_frames(spec)
    if frames:
        return frames[0]
    return _seed_input(ctx)


def _opts(ctx, name) -> dict:
    """Step options dict (enabled + arbitrary extras) for a step."""
    return ctx.config.step_options(name)


def _reconcile_with_input_entries(
    returned: pd.DataFrame,
    source: pd.DataFrame,
    entry_col: str = "Entry",
) -> pd.DataFrame:
    """Ensure every ``entry_col`` value in ``source`` survives in ``returned``.

    Several step implementations (``LigandRMSD``, ``SuperimposeStructures``,
    ``Vina``, ...) walk on-disk artifacts and quietly drop rows whose
    per-artifact processing raised. When a step's returned frame is missing
    entries that were in the input frame, downstream analysis loses those
    enzymes entirely.

    This helper re-introduces every missing ``entry_col`` value from
    ``source`` as a single row whose metric-side columns (those unique to
    ``returned``) are ``NaN``, while identifier/path columns coming from
    ``source`` are carried over unchanged. Column order matches
    ``returned``; extra source columns are appended.

    ``returned`` is returned unchanged when every ``source`` entry is
    already present.
    """
    import numpy as np

    if entry_col not in source.columns:
        return returned
    src_entries = source[entry_col].dropna().astype(str).unique()
    if len(src_entries) == 0:
        return returned
    if entry_col in returned.columns:
        got_entries = set(returned[entry_col].dropna().astype(str).unique())
    else:
        got_entries = set()
    missing = [e for e in src_entries if e not in got_entries]
    if not missing:
        return returned

    src_by_entry = source.drop_duplicates(subset=[entry_col]).set_index(
        source.drop_duplicates(subset=[entry_col])[entry_col].astype(str)
    )

    filler_rows = []
    for entry in missing:
        row = {c: np.nan for c in returned.columns}
        row[entry_col] = entry
        # Copy over identifier / path columns from source when available.
        if entry in src_by_entry.index:
            src_row = src_by_entry.loc[entry]
            for c in source.columns:
                if c in returned.columns and pd.isna(row[c]):
                    row[c] = src_row[c]
                elif c not in returned.columns:
                    row[c] = src_row[c]
        filler_rows.append(row)

    filler = pd.DataFrame(filler_rows)
    # Preserve returned's dtypes where possible by concatenating; new columns
    # from source (if any) are appended at the end.
    out = pd.concat([returned, filler], ignore_index=True, sort=False)
    return out


# --------------------------------------------------------------------------
# Docking phase
# --------------------------------------------------------------------------
def run_squidly(ctx, spec) -> pd.DataFrame:
    """Port of Docking._catalytic_residue_prediction (+ skip path)."""
    from structurezyme.utils.helpers import clean_protein_sequence, log_boxed_note
    from structurezyme.steps.squidly_step import Squidly

    opts = _opts(ctx, "squidly")
    df = _seed_input(ctx).copy()
    out_dir = _docking_dir(ctx)

    if opts.get("skip_catalytic_residue_prediction", False):
        df.to_pickle(out_dir / "squidly.pkl")
        return df

    df["Sequence"] = df["Sequence"].apply(clean_protein_sequence)

    squidly_step = Squidly(
        sequence_col="Sequence",
        id_col="Entry",
        model_size=opts.get("squidly_model_size", "3B"),
        as_threshold=opts.get("squidly_as_threshold", None),
        # Ensemble-mode residue-selection thresholds (squidly `run` CLI:
        # mean_prob default 0.6, mean_var default 0.225). Lower mean_prob /
        # higher mean_var => predicts lower-confidence residues, needed for
        # enzymes without a canonical catalytic triad. None => squidly default.
        mean_prob=opts.get("squidly_mean_prob", None),
        mean_var=opts.get("squidly_mean_var", None),
        num_threads=opts.get("squidly_num_threads") or ctx.config.runtime.num_threads,
    )
    df_squidly = squidly_step.execute(df)

    if "vina_residues" not in df_squidly.columns:
        df_squidly["vina_residues"] = None

    for col in ["Squidly_CR_Position", "vina_residues"]:
        if col not in df_squidly.columns:
            df_squidly[col] = ""

        def _norm(v):
            if v is None:
                return ""
            if isinstance(v, (list, tuple)):
                return "|".join(str(x).strip() for x in v if str(x).strip() != "")
            s = str(v).strip()
            return "" if s.lower() in ("nan", "none", "[]") else s

        df_squidly[col] = df_squidly[col].apply(_norm)

    use_vina = df_squidly["vina_residues"] != ""
    df_squidly["catalytic_residues"] = df_squidly["vina_residues"].where(
        use_vina, df_squidly["Squidly_CR_Position"]
    )

    # Enzymes without a canonical catalytic triad (and without user-specified
    # vina_residues) get an empty ``catalytic_residues``. We KEEP these rows:
    # chai/boltz co-fold the protein+ligand without needing residues, and the
    # downstream analysis (docking_metrics, superimposition, RMSD, geometric
    # filter, fpocket, SASA, PLIP) runs on those co-folded poses. Only vina is
    # skipped per-row for them (see run_vina), since it needs residues to place
    # its docking box. We notify the user which entries took this path.
    mask_empty = df_squidly["catalytic_residues"] == ""
    empty_entries = df_squidly.loc[mask_empty, "Entry"].tolist()
    if empty_entries:
        log_boxed_note(
            "No catalytic residues found (and no vina_residues specified) for: "
            + ", ".join(empty_entries)
            + ". These entries will be co-folded by chai/boltz and analysed "
            "downstream, but vina docking is skipped for them."
        )

    df_squidly.to_pickle(out_dir / "squidly.pkl")
    return df_squidly


def run_chai(ctx, spec) -> pd.DataFrame:
    """Port of Docking._run_chai."""
    from enzymetk.dock_chai_step import Chai
    from structurezyme.steps.save_step import Save

    out_dir = _docking_dir(ctx)
    num_threads = ctx.config.runtime.num_threads
    df_squidly = _first_input(ctx, spec)

    chai_dir = out_dir / "chai"
    chai_dir.mkdir(exist_ok=True, parents=True)
    if "cofactor_smiles" not in df_squidly.columns:
        df_squidly["cofactor_smiles"] = ""
    df_chai = df_squidly << (
        Chai("Entry", "Sequence", "substrate_smiles", "cofactor_smiles", chai_dir, num_threads)
        >> Save(out_dir / "chai.pkl")
    )
    df_chai.rename(columns={"output_dir": "chai_dir"}, inplace=True)
    return df_chai


def _boltz_together_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Route a together-mode frame for Boltz co-docking.

    substrate #0 stays the affinity binder in ``substrate_smiles``; substrate
    #1 (if any) is written to ``cofactor_smiles`` (docko ``id:C``). Enforces the
    2-substrate cap and rejects a 2nd substrate when a real cofactor already
    exists, because docko can only route one extra ligand SMILES safely.
    """
    out = df.copy()
    new_sub, new_cof = [], []
    existing = out["cofactor_smiles"] if "cofactor_smiles" in out.columns else None
    for pos, (_, row) in enumerate(out.iterrows()):
        subs = iter_substrates(row)
        if len(subs) > 2:
            raise ValueError(
                f"together-mode Boltz supports at most 2 substrates, got "
                f"{len(subs)} for Entry={row['Entry']!r}"
            )
        prior = "" if existing is None else str(existing.iloc[pos] or "")
        if prior in ("", "nan", "None"):
            prior = ""
        if len(subs) == 2 and prior:
            raise ValueError(
                f"together-mode Boltz cannot co-dock a 2nd substrate together "
                f"with a pre-existing cofactor for Entry={row['Entry']!r}"
            )
        new_sub.append(subs[0][0])
        new_cof.append(subs[1][0] if len(subs) == 2 else prior)
    out["substrate_smiles"] = new_sub
    out["cofactor_smiles"] = new_cof
    return out


def run_boltz(ctx, spec) -> pd.DataFrame:
    """Port of Docking._run_boltz."""
    from enzymetk.dock_boltz_step import Boltz
    from structurezyme.steps.save_step import Save

    opts = _opts(ctx, "boltz")
    out_dir = _docking_dir(ctx)
    num_threads = ctx.config.runtime.num_threads
    df_chai = _first_input(ctx, spec)

    if ctx.config.multi_substrate_mode == "together":
        df_chai = _boltz_together_frame(df_chai)

    boltz_dir = out_dir / "boltz/"
    boltz_dir.mkdir(exist_ok=True, parents=True)
    if "cofactor_smiles" not in df_chai.columns:
        df_chai["cofactor_smiles"] = None

    boltz_args = ["--cache", str(ctx.config.paths.boltz_cache_dir)]
    if opts.get("use_msa_server", True):
        boltz_args.append("--use_msa_server")
    # --no_kernels: disable optional CUDA fused kernels (require
    # cuequivariance_ops_torch); the pure-PyTorch fallback is always available.
    boltz_args.append("--no_kernels")

    df_boltz = df_chai << (
        Boltz("Entry", "Sequence", "substrate_smiles", "cofactor_smiles", boltz_dir,
              num_threads, args=boltz_args)
        >> Save(out_dir / "boltz.pkl")
    )
    df_boltz.rename(columns={"output_dir": "boltz_dir"}, inplace=True)
    return df_boltz


def _together_skips_vina(mode: str, df: pd.DataFrame) -> bool:
    """True when together-mode has any multi-substrate row (Vina can't co-dock)."""
    if mode != "together":
        return False
    return any(len(iter_substrates(row)) > 1 for _, row in df.iterrows())


def run_vina(ctx, spec) -> pd.DataFrame:
    """Port of Docking._run_vina (incl. AF2-missing fallback retry)."""
    from structurezyme.steps.dock_vina_step import Vina
    from structurezyme.utils.helpers import (
        delete_empty_subdirs,
        generate_boltz_structure_path,
        generate_chai_structure_path,
        log_boxed_note,
    )

    out_dir = _docking_dir(ctx)
    df_boltz = _first_input(ctx, spec)

    if _together_skips_vina(ctx.config.multi_substrate_mode, df_boltz):
        log_boxed_note(
            "Skipping vina docking (together-mode co-docking) for all rows; "
            "using chai/boltz co-folded poses."
        )
        df_boltz = df_boltz.copy()
        df_boltz["vina_dir"] = pd.NA
        df_boltz.to_pickle(out_dir / "vina.pkl")
        return df_boltz

    opts = _opts(ctx, "vina")
    num_threads = ctx.config.runtime.num_threads
    metagenomic_enzymes = opts.get("metagenomic_enzymes", 0)
    alt = opts.get("alternative_structure_for_vina", "Boltz")

    vina_dir = out_dir / "vina/"
    vina_dir.mkdir(exist_ok=True, parents=True)
    delete_empty_subdirs(vina_dir)

    # Per-row vina skip: vina needs catalytic_residues to place its docking box.
    # Rows with no residues (enzymes squidly found no catalytic triad for, and
    # with no user-specified vina_residues) are set aside here -- vina is not
    # run for them. They keep their chai_dir/boltz_dir and rejoin the frame at
    # the end with vina_dir = NaN, so docking_metrics falls back to the boltz
    # co-folded pose and the rest of the pipeline proceeds. See run_squidly.
    if "catalytic_residues" in df_boltz.columns:
        has_residues = df_boltz["catalytic_residues"].fillna("").astype(str).str.strip() != ""
    else:
        has_residues = pd.Series(True, index=df_boltz.index)
    df_skipped = df_boltz[~has_residues].copy()
    df_boltz = df_boltz[has_residues].reset_index(drop=True)
    if len(df_skipped):
        log_boxed_note(
            "Skipping vina docking (no catalytic residues) for: "
            + ", ".join(df_skipped["Entry"].astype(str).tolist())
            + ". These entries proceed with chai/boltz co-folded poses."
        )

    if len(df_boltz) == 0:
        # Nobody has residues -- vina does nothing; every row passes through.
        df_skipped["vina_dir"] = pd.NA
        df_skipped.to_pickle(out_dir / "vina.pkl")
        return df_skipped

    if metagenomic_enzymes == 1:
        if alt == "Chai":
            log_boxed_note("Fallback to Chai structures for docking due to missing AF2 structures.")
            df_boltz["structure"] = df_boltz["chai_dir"].apply(generate_chai_structure_path)
        elif alt == "Boltz":
            log_boxed_note("Fallback to Boltz structures for docking due to missing AF2 structures.")
            df_boltz["structure"] = df_boltz["boltz_dir"].apply(generate_boltz_structure_path)
    else:
        df_boltz["structure"] = None

    df_vina = df_boltz << (
        Vina("Entry", "structure", "Sequence", "substrate_smiles", "substrate_name",
             "catalytic_residues", vina_dir, num_threads)
    )
    df_vina.rename(columns={"output_dir": "vina_dir"}, inplace=True)

    if df_vina["vina_dir"].isnull().any():
        missing_entries = df_vina[df_vina["vina_dir"].isnull()]["Entry"].unique()
        delete_empty_subdirs(vina_dir)

        if alt == "Chai":
            log_boxed_note("Fallback to Chai structures for docking due to missing AF2 "
                           "structures. " + f"Entries: {list(missing_entries)}")
            df_missing = df_vina[df_vina["vina_dir"].isnull()].copy()
            df_missing["structure"] = df_missing["chai_dir"].apply(generate_chai_structure_path)
        elif alt == "Boltz":
            log_boxed_note("Fallback to Boltz structures for docking due to missing AF2 "
                           "structures. " + f"Entries: {list(missing_entries)}")
            df_missing = df_vina[df_vina["vina_dir"].isnull()].copy()
            df_missing["structure"] = df_missing["boltz_dir"].apply(generate_boltz_structure_path)

        df_missing_docked = df_missing << (
            Vina("Entry", "structure", "Sequence", "substrate_smiles", "substrate_name",
                 "catalytic_residues", vina_dir, num_threads)
        )
        df_missing_docked.rename(columns={"output_dir": "vina_dir_missing"}, inplace=True)

        df_vina_combined = pd.merge(
            df_vina, df_missing_docked[["Entry", "vina_dir_missing"]], on="Entry", how="left"
        )
        df_vina_combined["vina_dir"] = df_vina_combined.apply(
            lambda row: row["vina_dir"] if pd.notnull(row["vina_dir"]) else row["vina_dir_missing"],
            axis=1,
        )
        df_vina_combined.drop(columns=["vina_dir_missing"], inplace=True)
        df_vina = df_vina_combined.copy()

    # Rejoin the residue-less rows (vina_dir = NaN) so the full enzyme set flows
    # downstream. concat aligns on columns; skipped rows simply lack vina_dir.
    if len(df_skipped):
        if "vina_dir" not in df_skipped.columns:
            df_skipped["vina_dir"] = pd.NA
        df_vina = pd.concat([df_vina, df_skipped], ignore_index=True)

    df_vina.to_pickle(out_dir / "vina.pkl")
    return df_vina


def run_docking_metrics(ctx, spec) -> pd.DataFrame:
    """Port of Docking._extract_docking_quality_metrics.

    Consumes the most-downstream docking frame available: the vina checkpoint
    when vina ran, otherwise the boltz checkpoint (see registry inputs
    ``["boltz", "vina"]``). ``ctx.input_frames`` returns existing inputs in
    ``spec.inputs`` order, so the last available frame is the preferred one.
    """
    from structurezyme.steps.extract_docking_metrics_step import DockingMetrics
    from structurezyme.steps.save_step import Save

    out_dir = _docking_dir(ctx)
    run_vina = ctx.config.is_enabled("vina")

    frames = ctx.input_frames(spec)
    if frames:
        df = frames[-1].copy()
    else:
        df = _seed_input(ctx).copy()

    # Keep any row that has a usable docked structure. With per-row vina skip
    # (see run_vina), a row may legitimately have vina_dir = NaN but a valid
    # boltz_dir co-folded pose; such rows must survive. Rows with neither are
    # genuine failures and are dropped.
    if run_vina and "vina_dir" in df.columns:
        keep = df["vina_dir"].notna()
        if "boltz_dir" in df.columns:
            keep = keep | df["boltz_dir"].notna()
        df = df[keep].copy()
    else:
        df = df[df["boltz_dir"].notna()].copy()

    df_metrics = df << (
        DockingMetrics(input_dir=out_dir, output_dir=out_dir)
        >> Save(out_dir / "dockingmetrics.pkl")
    )
    return df_metrics


# --------------------------------------------------------------------------
# Superimposition phase
# --------------------------------------------------------------------------
def run_prepare_files(ctx, spec) -> pd.DataFrame:
    """Port of Superimposition._prepare_files_for_superimposition."""
    from structurezyme.steps.preparevina_step import PrepareVina
    from structurezyme.steps.preparechai_step import PrepareChai
    from structurezyme.steps.prepareboltz_step import PrepareBoltz

    docking_dir = _docking_dir(ctx)
    superimp_dir = _superimp_dir(ctx)
    include_vina = ctx.config.is_enabled("vina")

    df_metrics = pd.read_pickle(docking_dir / "dockingmetrics.pkl")
    preparedfiles_dir = superimp_dir / "preparedfiles_for_superimposition/"

    if include_vina:
        df_metrics << (
            PrepareVina("vina_dir", "substrate_name", preparedfiles_dir)
            >> PrepareChai("chai_dir", preparedfiles_dir, 1)
            >> PrepareBoltz("boltz_dir", preparedfiles_dir, 1)
        )
    else:
        df_metrics << (
            PrepareChai("chai_dir", preparedfiles_dir, 1)
            >> PrepareBoltz("boltz_dir", preparedfiles_dir, 1)
        )
    return df_metrics


def run_fastrelax(ctx, spec) -> pd.DataFrame:
    """Port of Superimposition._run_fastrelax."""
    from structurezyme.steps.fastrelax_step import FastRelax

    opts = _opts(ctx, "fastrelax")
    superimp_dir = _superimp_dir(ctx)
    num_threads = ctx.config.runtime.num_threads
    df_prep = _first_input(ctx, spec)

    fastrelax_dir = superimp_dir / "fastrelax"
    fastrelax_dir.mkdir(exist_ok=True, parents=True)
    step = FastRelax(
        output_dir=fastrelax_dir,
        mode=opts.get("fastrelax_mode", "ligand_focused"),
        top_k=opts.get("fastrelax_top_k", 2),
        drop_unrelaxed=opts.get("fastrelax_drop_unrelaxed", True),
        shell_radius=opts.get("fastrelax_shell_radius", 8.0),
        constraint_weight=opts.get("fastrelax_constraint_weight", 1.0),
        scorefunction=opts.get("fastrelax_scorefunction", "ref2015"),
        num_threads=num_threads,
    )
    df_relaxed = step.execute(df_prep)
    # Make the relaxed poses resolvable by the downstream analysis steps, which
    # look up preparedfiles_for_superimposition/<docked_structure>.pdb using the
    # now-relaxed pose names. See _publish_relaxed_to_prepared.
    _publish_relaxed_to_prepared(
        fastrelax_dir, superimp_dir / "preparedfiles_for_superimposition"
    )
    return df_relaxed


def _superimpose_input(ctx, spec) -> pd.DataFrame:
    """Select the frame superimposition operates on.

    Legacy ``Superimposition.run`` fed the *fastrelax-modified* frame into
    ``_superimposition`` when ``run_fastrelax=True``. In the modular DAG,
    fastrelax is an optional step, so we consume its checkpoint when the step
    is enabled *and* it actually produced one; otherwise we fall back to the
    ``prepare_files`` frame (the un-relaxed path). This keeps the default
    (fastrelax disabled) behavior identical while restoring the relaxed-input
    behavior when fastrelax runs.
    """
    if ctx.config.is_enabled("fastrelax"):
        fr = ctx.checkpoint_path("fastrelax")
        if fr.is_file():
            return pd.read_pickle(fr)
    prep = ctx.checkpoint_path("prepare_files")
    if prep.is_file():
        return pd.read_pickle(prep)
    return _seed_input(ctx)


def run_superimpose(ctx, spec) -> pd.DataFrame:
    """Port of Superimposition._superimposition (with fastrelax-aware input)."""
    from structurezyme.steps.superimposestructures_step import SuperimposeStructures
    from structurezyme.steps.save_step import Save
    from structurezyme.utils.helpers import valid_file_list

    superimp_dir = _superimp_dir(ctx)
    num_threads = ctx.config.runtime.num_threads
    include_vina = ctx.config.is_enabled("vina")
    df = _superimpose_input(ctx, spec)

    output_sup_dir = superimp_dir / "superimposed_structures"

    # Chai files are always required (chai is a mandatory step). Filter here
    # once so both the with-vina and without-vina branches share the guarantee.
    df = df[df["chai_files_for_superimposition"].apply(valid_file_list)]

    if include_vina:
        # Per-row vina skip: split rows into those that actually have valid
        # vina files vs. those that don't (e.g. residue-less enzymes for which
        # run_vina left vina_files_for_superimposition = None). The former get
        # the full 3-way superimpose (vina<->chai, vina<->boltz, chai<->boltz);
        # the latter get chai<->boltz only. Both subsets are concatenated so
        # every row survives into the downstream frame.
        has_vina = df["vina_files_for_superimposition"].apply(valid_file_list)
        df_with_vina = df[has_vina]
        df_without_vina = df[~has_vina]

        if len(df_with_vina):
            df_with_vina = df_with_vina << (
                SuperimposeStructures("vina_files_for_superimposition", "chai_files_for_superimposition",
                                      output_dir=output_sup_dir, name1="vina", name2="chai",
                                      num_threads=num_threads)
                >> SuperimposeStructures("vina_files_for_superimposition", "boltz_files_for_superimposition",
                                         output_dir=output_sup_dir, name1="vina", name2="boltz",
                                         num_threads=num_threads)
                >> SuperimposeStructures("chai_files_for_superimposition", "boltz_files_for_superimposition",
                                         output_dir=output_sup_dir, name1="chai", name2="boltz",
                                         num_threads=num_threads)
            )

        if len(df_without_vina):
            df_without_vina = df_without_vina << (
                SuperimposeStructures("chai_files_for_superimposition", "boltz_files_for_superimposition",
                                      output_dir=output_sup_dir, name1="chai", name2="boltz",
                                      num_threads=num_threads)
            )

        df_sup = pd.concat([df_with_vina, df_without_vina], ignore_index=True)
        df_sup.to_pickle(superimp_dir / "superimposedstructures.pkl")
    else:
        df_sup = df << (
            SuperimposeStructures("chai_files_for_superimposition", "boltz_files_for_superimposition",
                                  output_dir=output_sup_dir, name1="chai", name2="boltz",
                                  num_threads=num_threads)
            >> Save(superimp_dir / "superimposedstructures.pkl")
        )
    return df_sup


def run_protein_rmsd(ctx, spec) -> pd.DataFrame:
    """Port of Superimposition._proteinRMSD.

    The legacy method returns (pairwise, main); the pairwise frame is still
    written to disk, and the *main* frame is returned/checkpointed.
    """
    from structurezyme.steps.computeproteinRMSD_step import ProteinRMSD

    superimp_dir = _superimp_dir(ctx)
    df = _first_input(ctx, spec)

    proteinRMSD_dir = superimp_dir / "proteinRMSD"
    proteinRMSD_dir.mkdir(exist_ok=True, parents=True)
    input_dir = superimp_dir / "superimposed_structures"
    df_proteinRMSD_pairwise, df_proteinRMSD = df << (
        ProteinRMSD("Entry", input_dir=input_dir, output_dir=proteinRMSD_dir,
                    visualize_heatmaps=True)
    )
    df_proteinRMSD_pairwise.to_pickle(superimp_dir / "proteinRMSD_pairwise.pkl")
    df_proteinRMSD.to_pickle(superimp_dir / "proteinRMSD.pkl")
    return df_proteinRMSD


def run_ligand_rmsd(ctx, spec) -> pd.DataFrame:
    """Port of Superimposition._ligandRMSD.

    Returns/checkpoints the metrics-augmented main frame; writes pairwise and
    the fixed-name ``ligandRMSD.pkl`` that geometric filtering consumes.
    """
    from structurezyme.steps.computeligandRMSD_step import LigandRMSD
    from structurezyme.utils.helpers import extract_docking_metrics

    opts = _opts(ctx, "ligand_rmsd")
    superimp_dir = _superimp_dir(ctx)
    max_matches = opts.get("max_matches", 1000)
    df = _first_input(ctx, spec)

    ligandRMSD_dir = superimp_dir / "ligandRMSD"
    ligandRMSD_dir.mkdir(exist_ok=True, parents=True)
    input_dir = superimp_dir / "superimposed_structures"

    df_ligandRMSD_pairwise, df_ligandRMSD = df << (
        LigandRMSD("Entry", input_dir=input_dir, output_dir=ligandRMSD_dir,
                   visualize_heatmaps=True, maxMatches=max_matches)
    )
    # LigandRMSD silently drops entries whose pose comparisons all failed
    # (rdkit valence / substructure errors). Reintroduce those entries with
    # NaN metric columns so downstream analysis still sees the enzyme.
    df_ligandRMSD = _reconcile_with_input_entries(df_ligandRMSD, df)
    df_ligandRMSD.to_pickle(superimp_dir / "ligandRMSD_prior.pkl")
    df_ligandRMSD_w_metrics = extract_docking_metrics(df_ligandRMSD)
    df_ligandRMSD_w_metrics = _reconcile_with_input_entries(
        df_ligandRMSD_w_metrics, df
    )
    df_ligandRMSD_w_metrics.to_pickle(superimp_dir / "ligandRMSD.pkl")
    df_ligandRMSD_pairwise.to_pickle(superimp_dir / "ligandRMSD_pairwise.pkl")
    return df_ligandRMSD_w_metrics


# --------------------------------------------------------------------------
# Geometric filtering phase
# --------------------------------------------------------------------------
def run_geometric_filter(ctx, spec) -> pd.DataFrame:
    """Port of GeometricFilters._run_geometric_filtering."""
    from structurezyme.steps.geometric_filtering_cofactor_MCS import GeneralGeometricFiltering
    from structurezyme.steps.geometric_filtering_esterase import EsteraseGeometricFiltering
    from structurezyme.steps.save_step import Save

    opts = _opts(ctx, "geometric_filter")
    superimp_dir = _superimp_dir(ctx)
    geo_dir = _geo_dir(ctx)
    esterase = opts.get("esterase", 0)
    df = _first_input(ctx, spec)

    prepared = superimp_dir / "preparedfiles_for_superimposition"
    if esterase == 1:
        df_geo_filter = df << (
            EsteraseGeometricFiltering(preparedfiles_dir=prepared, output_dir=geo_dir)
            >> Save(geo_dir / "geometricfiltering.pkl")
        )
    else:
        df_geo_filter = df << (
            GeneralGeometricFiltering(preparedfiles_dir=prepared, output_dir=geo_dir)
            >> Save(geo_dir / "geometricfiltering.pkl")
        )
    # Reconcile: preserve every input Entry so downstream analysis still sees
    # enzymes whose geometric filter dropped all poses.
    return _reconcile_with_input_entries(df_geo_filter, df)


def run_fpocket(ctx, spec) -> pd.DataFrame:
    """Port of GeometricFilters._active_site_volume."""
    from structurezyme.steps.fpocket_step import Fpocket
    from structurezyme.steps.save_step import Save

    superimp_dir = _superimp_dir(ctx)
    geo_dir = _geo_dir(ctx)
    df = _first_input(ctx, spec)

    prepared = superimp_dir / "preparedfiles_for_superimposition"
    fpocket_dir = geo_dir / "ASVolume"
    fpocket_dir.mkdir(exist_ok=True, parents=True)
    df_ASVolume = df << (
        Fpocket(preparedfiles_dir=prepared, output_dir=fpocket_dir)
        >> Save(geo_dir / "ASvolume.pkl")
    )
    return _reconcile_with_input_entries(df_ASVolume, df)


def run_ligand_sasa(ctx, spec) -> pd.DataFrame:
    """Port of GeometricFilters._ligand_surface_exposure."""
    from structurezyme.steps.ligandSASA_step import LigandSASA
    from structurezyme.steps.save_step import Save

    superimp_dir = _superimp_dir(ctx)
    geo_dir = _geo_dir(ctx)
    df = _first_input(ctx, spec)

    prepared = superimp_dir / "preparedfiles_for_superimposition"
    ligandSASA_dir = geo_dir / "LigandSASA"
    df_ligandSASA = df << (
        LigandSASA(input_dir=prepared, output_dir=ligandSASA_dir)
        >> Save(geo_dir / "ligandSASA.pkl")
    )
    return _reconcile_with_input_entries(df_ligandSASA, df)


def run_plip(ctx, spec) -> pd.DataFrame:
    """Port of GeometricFilters._plip_interactions.

    Writes the fixed-name ``structural_features_final.pkl`` that PLACER
    consumes, matching the legacy ``GeometricFilters.run`` finalization.
    """
    from structurezyme.steps.plip_step import PLIP
    from structurezyme.steps.save_step import Save

    superimp_dir = _superimp_dir(ctx)
    geo_dir = _geo_dir(ctx)
    df = _first_input(ctx, spec)

    prepared = superimp_dir / "preparedfiles_for_superimposition"
    df_plip = df << (
        PLIP(input_dir=prepared, output_dir=geo_dir)
        >> Save(geo_dir / "plip_interactions.pkl")
    )
    df_plip = _reconcile_with_input_entries(df_plip, df)
    df_plip.to_pickle(geo_dir / "structural_features_final.pkl")
    return df_plip


# --------------------------------------------------------------------------
# PLACER (opt-in)
# --------------------------------------------------------------------------
def run_placer(ctx, spec) -> pd.DataFrame:
    """Port of the PLACER block in Pipeline.run."""
    from structurezyme.steps.PLACER_step import PLACER

    opts = _opts(ctx, "placer")
    superimp_dir = _superimp_dir(ctx)
    geo_dir = _geo_dir(ctx)
    num_threads = ctx.config.runtime.num_threads

    predict_ligand = opts.get("placer_predict_ligand", None)
    if predict_ligand is None:
        raise ValueError(
            "placer enabled but placer_predict_ligand is not set (e.g. 'A-HEM-154')"
        )

    geo_pkl = geo_dir / "structural_features_final.pkl"
    df_geo = pd.read_pickle(geo_pkl)
    placer = PLACER(
        preparedfiles_dir=superimp_dir / "preparedfiles_for_superimposition",
        output_dir=ctx.layout.root / "placer",
        predict_ligand=predict_ligand,
        placer_env_path=opts.get("placer_env_path", ctx.config.paths.placer_env_path),
        nsamples=opts.get("placer_nsamples", 50),
        rerank=opts.get("placer_rerank", "prmsd"),
        num_threads=num_threads,
    )
    df_placer = placer.execute(df_geo)
    df_placer = _reconcile_with_input_entries(df_placer, df_geo)
    df_placer.to_pickle(geo_pkl)
    return df_placer
