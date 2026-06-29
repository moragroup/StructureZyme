# Lab Filterzyme Sanity Run — Working Record

**Owner:** lherrmann (LCH)
**Started:** 2026-06-26
**Status:** end-to-end-ish run achieved. Pipeline cleared all of Docking + most of Superimposition. Breaks at Stage B.3 ligand RMSDs with `KeyError: 'docked_structure'` (new finding, not in bug catalogue).

## Files in this folder

| File | Purpose | Audience |
|------|---------|----------|
| `00_index.md` | This file | everyone |
| `01_journal.md` | Chronological log of every install / fix / decision, with revert path for each | future me + anyone reproducing |
| `02_PI_status.md` | One-pager for Ariane: what works, what's broken, what we need from her | Ariane / PI |
| `03_env_state.md` | Snapshot of the env we built, versions, divergences from Ariane's README | future me |

## Quick orientation

We are executing the plan at `.kilo/plans/1782378970772-lab-filterzyme-sanity-run.md`. That plan assumed we could use Ariane's pre-built `filterzyme` conda env. **That assumption was wrong** — her env is private to her account. We pivoted to building a local env per her README, and have been patching missing system/conda packages as the pipeline reveals them.

Six environment patches later (cu128 torch, pdbfixer/openbabel/plip via conda, numpy pin <2.2, regex upgrade, lab docko 0.1.5 from a copy at `~/docko_lab_LCH`) the pipeline runs end-to-end through Chai → Boltz → Vina → DockingMetrics → Superimpose → **proteinRMSD (past Ariane's predicted break point)**. It then breaks at ligandRMSD with a `KeyError: 'docked_structure'` raised in `helpers.py:319`, preceded by `Error selecting best docked structures: 'Entry'`. This is a new bug, not in `05_bugs_by_severity.md`.
