# 11. Code Walkthrough â "What happens where"

Function-level English summary of every important callsite, with `file:line` citations. Read top-to-bottom this is the same order in which a single entry passes through the pipeline. Both v1 and v2 are walked; v2 is the canonical version.

## Top-level: `Pipeline.run`

### v2 (canonical)

`pipeline_v2.py:344` â `Pipeline.run(self)`:

1. Calls `Docking(self.df, ..., self.base_output_dir).run()` and binds the returned `(df_best_dock, df_dock_metrics)`.
2. Calls `Superimposition(df_best_dock, ..., self.base_output_dir).run()` and binds `(df_best_struct, df_struct_metrics)`.
3. Calls `GeometricFilters(df_best_struct, ..., self.base_output_dir).run()` and binds `df_final`.
4. Pickles `df_final` to `<base_output_dir>/geometricfiltering/structural_features_final.pkl`.

### v1

`pipeline.py:355` â `Pipeline.run`: same structure, single-DataFrame returns from `_proteinRMSD` and `_ligandRMSD`, brittle `Path(...).parent` path at `pipeline.py:265`.

## Stage 1 â `Docking`

### `Docking.__init__`

`pipeline_v2.py:42` â accepts `df`, `output_dir`, `num_threads`, `squidly_dir`, `esm2_model`, `skip_catalytic_residue_prediction`, `metagenomic_enzymes`, `alternative_structure_for_vina`. Default `alternative_structure_for_vina='Chai'`. (`pipeline.py:38` has the same signature plus the typo'd attribute name internally.)

### `Docking._catalytic_residue_prediction`

`pipeline_v2.py:78` (v1: `pipeline.py:71`).

Walkthrough:

1. `clean_protein_sequence` (`helpers.py:125`) strips ambiguous residues from every `Sequence`.
2. Deduplicate by `Sequence` â `pred_in`.
3. Run Squidly. v2: `df_cat_res = pred_in << ActiveSitePred('Entry', 'Sequence')` â direct import from `enzymetk`. v1: `pred_in << ActiveSitePred('Entry', 'Sequence', self.squidly_dir, self.num_threads)` â local subprocess-wrapped predictor.
4. v2 reads `df_cat_res.id` and `df_cat_res.residues` (`pipeline_v2.py:87`). v1 reads `df_cat_res.label` and `df_cat_res.Squidly_CR_Position` (`pipeline.py:80-82`). These schemas are NOT interchangeable â see P0-4.
5. Merge predictions back into `self.df` â `df_squidly`.
6. If the user supplied `vina_residues`, combine with Squidly's via pipe-delimited string; otherwise use only Squidly's. Drop entries where neither is present.
7. Return `df_squidly` with a `catalytic_residues` column.

### `Docking._run_chai`

`pipeline_v2.py:127` (v1: `pipeline.py:124`).

1. If `cofactor_smiles` is not a column, fill with `''` (v1) or `None` (v2 also `''` initially, then `None` later â the sentinel mismatch lives in v1 only at this depth, P2-2).
2. `df_chai = df_squidly << (Chai('Entry','Sequence','substrate_smiles','cofactor_smiles', output_dir=...) >> Save(...))`.
3. Returns `df_chai` carrying a per-entry path to the predicted CIF.

### `Docking._run_boltz`

`pipeline_v2.py:138` (v1: `pipeline.py:135`). Mirror of `_run_chai` for Boltz-2. Note the v1 sentinel asymmetry mentioned in P2-2.

### `Docking._run_vina`

`pipeline_v2.py:150` (v1: `pipeline.py:144`). The most complex method in the file.

1. If `metagenomic_enzymes == 0`: try Vina with AF2 receptor (`dock_vina_step.py:44 get_alphafold_structure`). On failure, fall back to Chai or Boltz prediction based on `alternative_structure_for_vina`.
2. If `metagenomic_enzymes == 1`: skip AF2 entirely; go straight to the Chai/Boltz fallback.
3. **v1 BUG (P0-1):** lines 151 and 154 read `self.alternative_strucuture_for_vina` (sic). The attribute is `alternative_structure_for_vina` at line 50. `AttributeError` when this branch is reached.
4. v2: lines 158, 161, 178, 186 use the correct spelling.
5. Outputs flow to `<output_dir>/vina/<Entry>/`.

### `Docking._extract_docking_quality_metrics`

`pipeline_v2.py:210`. Constructs a `DockingMetrics(input_dir=..., output_dir=...)` and runs it.

### `DockingMetrics.execute`

`extract_docking_metrics_step.py:252`. For every entry directory:

1. Parse the Vina log `.txt` for affinity scores per pose. Hard-coded log-format assumption.
2. Parse the Chai `.npz` for ptm/iptm/plddt.
3. Parse the Boltz `.json` for confidence + affinity. Hard-coded keys at lines 230â242: `affinity_pred_value`, `affinity_pred_value1`, `affinity_pred_value2`, etc. Breaks silently if Boltz upstream renames.
4. On any exception: `print` at line 245, return an empty dict â silent failure (P2-5).
5. Return one DataFrame row per entry with dict-valued metric columns.

### `Docking.run`

`pipeline_v2.py:329`. Composes the above in sequence and returns `(df_best_dock, df_dock_metrics)`. v2 uses `@log_usage` (`helpers.py:68`) for elapsed-time and resident-memory tracking on every stage.

## Stage 2 â `Superimposition`

### `Superimposition.__init__`

`pipeline_v2.py:236`. Accepts `df`, `output_dir`, etc.

### `Superimposition._prepare_files`

Per-tool `PrepareVina` (`preparevina_step.py:163`), `PrepareChai` (`preparechai_step.py:155`), `PrepareBoltz` (`prepareboltz_step.py:64`) flatten the per-tool output directories into a uniform layout under `<output_dir>/superimposed_structures/<Entry>/`.

### `Superimposition._superimposition`

`pipeline_v2.py:248` (v1: `pipeline.py:240`). Three pairwise calls: vina-chai, vina-boltz, chai-boltz. Each backed by `SuperimposeStructures(Step)` at `superimposestructures_step.py:236`, which wraps biotite's `superimpose_homologs`.

### `Superimposition._proteinRMSD`

`pipeline_v2.py:262` (v1: `pipeline.py:252`). Calls `ProteinRMSD(Step)` (`computeproteinRMSD_step.py:177`).

- v2 returns `(df_proteinRMSD_pairwise, df_proteinRMSD)` â a 2-tuple (`pipeline_v2.py:269`).
- v1 returns `df_proteinRMSD` â single DataFrame.

This is the cross-version API incompatibility called out in P0-2.

### `Superimposition._ligandRMSD`

`pipeline_v2.py:272` (v1: `pipeline.py:260`). Calls `LigandRMSD(Step)` (`computeligandRMSD_step.py:370`). Then:

- v2: enriches via `extract_docking_metrics` (`helpers.py:277`) â lookup the per-pose Vina/Chai/Boltz scalars into the best-structures DataFrame.
- v1: calls `add_metrics_to_best_structures` reading from the brittle path `Path(self.output_dir).parent / 'docking/dockingmetrics.pkl'` (`pipeline.py:265`, P1-6). Internally `add_metrics_to_best_structures` calls `add_metrics` (`helpers.py:221`), which is **broken** (P0-3).

### `extract_docking_metrics` (the working one)

`helpers.py:277`. Given `(best_structures_df, dockmetrics_df)`:

1. Pulls per-pose Vina affinities out of the `vina_affinities` dict column on `dockmetrics_df`, keyed by the Vina pose index in `best_structures_df.docked_structure`.
2. Pulls per-pose Chai/Boltz scalars out of `dictâ’columns` (assembled from the column lists `chai_columns` and `boltz_columns` defined locally at lines 290 and 295).
3. Returns the best-structures DataFrame enriched with one column per metric.

This function is the template that `add_metrics` was clearly meant to be â see P0-3.

## Stage 3 â `GeometricFilters`

### `GeometricFilters.__init__`

`pipeline_v2.py:285` (v1: `pipeline.py:271`). Accepts `df`, `output_dir`, `esterase` (int flag).

### `GeometricFilters._geometric_filtering`

`pipeline_v2.py:295` (approx). Dispatches:

- `esterase == 1` â `EsteraseGeometricFiltering(Step)` (`geometric_filtering_esterase.py:418`).
- `esterase == 0` â `GeneralGeometricFiltering(Step)` (`geometric_filtering_cofactor_MCS.py:435` in v2; `geometric_filtering_cofactor.py:450` in v1).

The chosen filter computes per-pose: catalytic-residue / cofactor distances, SMARTS-anchored substrate orientation, optional closest-nucleophile geometry. Returns a per-pose DataFrame with the geometric features as columns.

### `GeometricFilters._fpocket`

`pipeline_v2.py:309` (approx). Constructs `Fpocket(Step)` (`fpocket_step.py:232`) and runs it. Shells out to the `fpocket` binary; parses the pocket .info output; returns volume, druggability score, hydrophobicity per pose.

### `GeometricFilters._ligandSASA`

Constructs `LigandSASA(Step)` (`ligandSASA_step.py:81`). Uses `SingleLigandSelect` (`helpers.py`) to extract the substrate chain, calls FreeSASA, returns absolute and relative SASA.

### `GeometricFilters._plip`

Constructs `PLIP(Step)` (`plip_step.py:81`). Runs the PLIP Python API; returns counts of hydrogen bonds, salt bridges, hydrophobic contacts, pi-stacking, pi-cation, halogen bonds, water bridges.

### `GeometricFilters.run`

Merges all four scorers' outputs into a single per-pose `df_final` and returns it for the top-level pickle.

## Step base classes

`step.py:6` â `Pipeline` class with `__rshift__` (`>>`) and `__rlshift__` (`<<`) operators. `pipe << df` triggers `pipe.execute(df)`. `step1 >> step2` returns a `Pipeline` that calls them in sequence.

`step.py:51` â first `Step` class (with placeholder docstring at line 54).

`step.py:66` â second `Step` class, shadows the first. Both implement the operator dunders so behaviour is preserved. CQ-2 is to delete the first.

## Utility highlights

- `helpers.py:45-58` â `log_section`, `log_subsection`, `log_boxed_note`: ASCII-art section headers for the run log.
- `helpers.py:68` â `log_usage` decorator: time + RSS memory around a stage.
- `helpers.py:105` / `helpers.py:115` â `generate_boltz_structure_path` / `generate_chai_structure_path`: hard-coded directory-layout encoders for the upstream tools. Brittle (CQ-15).
- `helpers.py:125` â `clean_protein_sequence`: strips ambiguous amino-acid codes.
- `helpers.py:221` â `add_metrics`: **broken** (P0-3).
- `helpers.py:277` â `extract_docking_metrics`: working sibling, template for the fix.
- `helpers.py:410-484` â chain-extraction helpers, duplicated downstream (CQ-1).
