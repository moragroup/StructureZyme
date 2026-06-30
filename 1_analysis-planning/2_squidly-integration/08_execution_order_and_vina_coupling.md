# 8. Execution Order and Vina Coupling

Short reference for three questions that came up after the initial plan was drafted.

## Q: Does Squidly always run, or is there still a skip flag?

**Skip flag still exists and we keep it.** `Docking.__init__` accepts
`skip_catalytic_residue_prediction: bool = False` (`pipeline_v2.py:56`) and the run
method short-circuits at `pipeline_v2.py:76`:

```
if self.skip_catalytic_residue_prediction:
    log_section("Skipping catalytic residue prediction")
    df_squidly = self.df
else:
    df_squidly = self._catalytic_residue_prediction()
```

Behaviour in the new `Squidly(Step)` design:

- Flag is preserved with the same name and default (`False`).
- When skipped, `catalytic_residues` is populated **only** from user-supplied
  `vina_residues` (if present). If neither Squidly ran nor `vina_residues` was given,
  the entry is dropped before Vina, matching today's behaviour at
  `pipeline_v2.py:126-133`.
- Skipping is only sensible when (a) `run_vina=False`, or (b) `vina_residues` is
  provided by the user for every entry. The new step will log a clear warning when
  it detects skipped-and-Vina-enabled without user residues.

## Q: Does Squidly run before everything else?

**Yes, Squidly is the first stage.** Order inside `Docking.run` is fixed
(`pipeline_v2.py:74-92`):

```
1. _catalytic_residue_prediction   (Squidly)         <-- first
2. _run_chai                       (Chai-1)
3. _run_boltz                      (Boltz-2)
4. _run_vina                       (Vina, optional)
5. _extract_docking_quality_metrics
```

It must run first because its output column `catalytic_residues` is consumed by Vina
later in the same stage. Chai and Boltz do not read it.

The new design does not change this ordering. The only refactor is *what* runs inside
step 1 (the new `Squidly(Step)` class instead of `enzymetk.ActiveSitePred` plus the
column-rename glue).

## Q: Is Squidly used to bias Vina docking?

**Yes, directly.** Vina is called with `catalytic_residues` as a positional argument
(`pipeline_v2.py:192` and the retry at `pipeline_v2.py:210`):

```
df_vina = df_boltz << (Vina('Entry', 'structure', 'Sequence',
                            'substrate_smiles', 'substrate_name',
                            'catalytic_residues',           <-- Squidly output
                            vina_dir, self.num_threads))
```

Inside `filterzyme/steps/dock_vina_step.py`, those residues are forwarded to
`docko.dock(..., residues=residues)`. `docko` computes the geometric centroid of those
CA atoms and places the 10x10x10 Angstrom Vina search box on it (box size hard-coded;
see `04_model_integration_audit.md` row 3 for the Vina audit). So:

- Squidly's residue list -> centroid -> Vina search box centre.
- User-supplied `vina_residues` takes precedence over Squidly's prediction
  (`pipeline_v2.py:136-137`: `df_squidly['vina_residues'].where(use_vina, df_squidly['Squidly_CR_Position'])`).
- Chai and Boltz are **not** biased by Squidly. They predict the full complex from
  sequence + ligand SMILES and ignore the residue list.

This is why disabling Squidly with `run_vina=True` and no `vina_residues` is a misuse:
Vina has no pocket to dock into. The new step's warning (above) makes this loud.

## Summary diagram

```
      input DataFrame
            |
            v
   +-------------------+
   | 1. Squidly (NEW)  |  <-- skip_catalytic_residue_prediction toggles this off
   +-------------------+
            |
            | catalytic_residues column added
            v
   +-------------------+     +-------------------+
   | 2. Chai-1         |     | 3. Boltz-2        |   (neither reads catalytic_residues)
   +-------------------+     +-------------------+
            \                       /
             \                     /
              v                   v
            +-----------------------+
            | 4. Vina (optional)    |  <-- reads catalytic_residues to place search box
            +-----------------------+
                       |
                       v
            +-----------------------+
            | 5. DockingMetrics     |
            +-----------------------+
```
