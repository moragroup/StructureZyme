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

    mask_empty = (df_squidly["Squidly_CR_Position"] == "") & (df_squidly["vina_residues"] == "")
    empty_entries = df_squidly.loc[mask_empty, "Entry"].tolist()
    if empty_entries:
        log_boxed_note(
            "Removing entries without catalytic residues and without specified "
            "residues for vina docking: " + ", ".join(empty_entries)
        )
    df_squidly = df_squidly[~mask_empty].reset_index(drop=True)

    use_vina = df_squidly["vina_residues"] != ""
    df_squidly["catalytic_residues"] = df_squidly["vina_residues"].where(
        use_vina, df_squidly["Squidly_CR_Position"]
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


def run_boltz(ctx, spec) -> pd.DataFrame:
    """Port of Docking._run_boltz."""
    from enzymetk.dock_boltz_step import Boltz
    from structurezyme.steps.save_step import Save

    opts = _opts(ctx, "boltz")
    out_dir = _docking_dir(ctx)
    num_threads = ctx.config.runtime.num_threads
    df_chai = _first_input(ctx, spec)

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


def run_vina(ctx, spec) -> pd.DataFrame:
    """Port of Docking._run_vina (incl. AF2-missing fallback retry)."""
    from structurezyme.steps.dock_vina_step import Vina
    from structurezyme.utils.helpers import (
        delete_empty_subdirs,
        generate_boltz_structure_path,
        generate_chai_structure_path,
        log_boxed_note,
    )

    opts = _opts(ctx, "vina")
    out_dir = _docking_dir(ctx)
    num_threads = ctx.config.runtime.num_threads
    metagenomic_enzymes = opts.get("metagenomic_enzymes", 0)
    alt = opts.get("alternative_structure_for_vina", "Boltz")
    df_boltz = _first_input(ctx, spec)

    vina_dir = out_dir / "vina/"
    vina_dir.mkdir(exist_ok=True, parents=True)
    delete_empty_subdirs(vina_dir)

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

    if run_vina and "vina_dir" in df.columns:
        df = df[df["vina_dir"].notna()].copy()
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
        ligand_resname=opts.get("ligand_resname", "LIG"),
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

    if include_vina:
        df = df[df["vina_files_for_superimposition"].apply(valid_file_list)]
        df = df[df["chai_files_for_superimposition"].apply(valid_file_list)]

        df_sup = df << (
            SuperimposeStructures("vina_files_for_superimposition", "chai_files_for_superimposition",
                                  output_dir=output_sup_dir, name1="vina", name2="chai",
                                  num_threads=num_threads)
            >> SuperimposeStructures("vina_files_for_superimposition", "boltz_files_for_superimposition",
                                     output_dir=output_sup_dir, name1="vina", name2="boltz",
                                     num_threads=num_threads)
            >> SuperimposeStructures("chai_files_for_superimposition", "boltz_files_for_superimposition",
                                     output_dir=output_sup_dir, name1="chai", name2="boltz",
                                     num_threads=num_threads)
            >> Save(superimp_dir / "superimposedstructures.pkl")
        )
    else:
        df = df[df["chai_files_for_superimposition"].apply(valid_file_list)]

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
    df_ligandRMSD.to_pickle(superimp_dir / "ligandRMSD_prior.pkl")
    df_ligandRMSD_w_metrics = extract_docking_metrics(df_ligandRMSD)
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
    return df_geo_filter


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
    return df_ASVolume


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
    return df_ligandSASA


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
    df_placer.to_pickle(geo_pkl)
    return df_placer
