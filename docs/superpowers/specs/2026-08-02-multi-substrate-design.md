# Multi-Substrate Support — Design

Date: 2026-08-02
Status: Approved (design), ready for implementation plan

## Problem

The pipeline schema supports exactly **one** `substrate_smiles` per `Entry`
row. Every downstream step reads a single scalar `row['substrate_smiles']`
(and `substrate_name` / `substrate_moiety`), and the Chai/Boltz engines write
outputs keyed on `Entry` alone. This blocked the halogenase run
(pyridine + tryptophan), which needs two substrates per enzyme.

Two distinct biological use-cases exist, and the user must choose per run:

1. **Screening ("separate")** — it is unknown which substrate the enzyme
   acts on; dock each substrate independently on the enzyme and compare.
2. **Co-docking ("together")** — both substrates occupy the active site
   simultaneously (e.g. two substrates forming one product).

## Goal

Add a per-run `multi_substrate_mode` flag supporting `off` (default,
unchanged), `separate`, and `together`, without breaking the existing
one-row-per-Entry pipeline or the current 197-test baseline.

## Engine feasibility (verified against docko source)

Source read at `/mnt/storage01/home/lherrmann/docko_lab_LCH/docko`
(the editable install actually used by the `filterzyme` env).

| Engine | 2 substrates + 1 cofactor? | How |
|--------|----------------------------|-----|
| **Chai** | Yes | `run_chai` (chai.py:64-70) splits `substrate_smiles` on `.` into one `>ligand-substrate-i` FASTA record each; cofactors are list-capable (chai.py:51-62). Protein + N substrates + M cofactors all fold together. |
| **Boltz** | Yes | `run_boltz_affinity` -> `write_yaml` (boltz.py:27-105) emits the primary substrate as ligand `id:B`; additional ligand entities become `id:C, D, …`. Affinity is scored for `B` only. |
| **Vina** | No | `dock` (docko.py:47-59) + `format_ligand` (helpers.py:290-320) build ONE RDKit molecule from ONE SMILES; `"A.B"` becomes a single two-fragment blob. No cofactor slot. |

Consequences baked into the design:
- **Chai** needs no code change for co-docking (native `.` split).
- **Boltz** needs substrate #2 routed as an extra ligand entity (like a
  cofactor) so it is a real second ligand, not a two-fragment `B`.
- **Vina** cannot co-dock; it is auto-skipped in `together` mode.

## Input schema & delimiters

Multi-substrate values are packed into the existing columns:

- `substrate_smiles` — one or more SMILES joined by `.` (matches docko's
  native Chai split, e.g. `"c1ccncc1.C(C(=O)O)N"`).
- `substrate_name` — parallel list joined by `|` (e.g. `"pyridine|tryptophan"`).
- `substrate_moiety` — parallel list joined by `|`, aligned by position.
- `cofactor_smiles` / `cofactor_moiety` — unchanged (optional; docko already
  list-capable).

Rationale for delimiters: `.` is the SMILES fragment separator Chai already
splits on; `|` is used for `substrate_name`/`substrate_moiety` because `.` and
`,` occur inside SMILES/SMARTS.

New config field on `RunConfig`:

- `multi_substrate_mode: "off" | "separate" | "together"` (default `"off"`).

## Mode semantics

### `off` (default)
Today's behavior, untouched. One SMILES per row. Full backward compatibility.

### `separate` (screening)
Row expansion happens once, at seeding (`Runner._seed_and_validate`). A row
with >1 substrate expands into one row per substrate:

```
in:  Entry=P12345               substrate_smiles="pyr.trp"
out: Entry=P12345__s0 enzyme_id=P12345 substrate_smiles="pyr" substrate_name="pyridine"
     Entry=P12345__s1 enzyme_id=P12345 substrate_smiles="trp" substrate_name="tryptophan"
```

- `enzyme_id` (new column) preserves grouping; `Entry` becomes the unique key.
- Downstream steps are **unchanged** — each still sees one substrate per Entry.
- Output paths never collide (the suffix is in every Entry-keyed path).
- Single-substrate rows are **not** expanded/suffixed (Entry unchanged).

### `together` (co-docking)
No row expansion; the row stays intact with dot/pipe-joined lists.

- **Chai**: `substrate_smiles` passed through; docko splits natively.
- **Boltz**: substrate #1 -> ligand `B`; substrate #2 (+ cofactor) routed as
  additional ligand entities.
- **Vina**: auto-skipped (per-row skip + log) when >1 substrate.
- **Downstream analysis**: loops over both substrates, emits per-substrate
  suffixed columns (`_s0`, `_s1`); does **not** hard-filter.

## Components (where each change lives)

Mode logic is concentrated in two narrow places; analysis-step changes are
mechanical and inert unless a row genuinely has >1 substrate.

1. **Config** — `structurezyme/config.py`: add `multi_substrate_mode` field +
   validation; add to `cli._template`. `REQUIRED_INPUT_COLUMNS` in
   `runner.py` still requires `substrate_smiles`.

2. **Seeding / expansion** — `structurezyme/runner.py :: _seed_and_validate`
   plus a new helper `_expand_substrates(df, mode)`. The **only** place that
   knows about `separate` row expansion and delimiter parsing. Validates that
   the `.`/`|` list lengths align; fails fast (naming the Entry) before GPU
   work. Resume-safe: an existing `_input.pkl` is never re-expanded.

3. **Shared substrate iterator** — new `iter_substrates(row)` helper (e.g. in
   `structurezyme/utils/helpers.py`) yielding an ordered list of
   `(smiles, name, moiety)`. For single-substrate rows it yields exactly one
   tuple, so consumers emit **unsuffixed** columns identical to today.

4. **Docking feed (together only)** — the Boltz step wiring
   (`step_runners.py` / `pipeline.py` where `Boltz(...)` is built) routes
   substrate #2 as an extra ligand entity. Chai: no change. Vina: add a
   together-mode >1-substrate skip guard (reuses the existing per-row skip).

5. **Analysis steps (together only)** — `geometric_filter`, `ligand_rmsd`,
   `plip`, `ligand_sasa`, `fpocket` each loop over `iter_substrates(row)` and
   write `_s{i}`-suffixed columns. In `off`/`separate` mode the loop runs once
   and columns stay unsuffixed.

   `ligand_rmsd` additionally emits a **joint** metric in `together` mode:
   after protein superposition it chemically disambiguates the two co-docked
   substrates via `closest_ligands_by_element_composition`, computes each
   substrate's own cross-method RMSD (`ligand_rmsd_s0`, `ligand_rmsd_s1`) AND
   a `ligand_rmsd_joint` over both substrates as one point set. Per-substrate
   captures each ligand's pose confidence; joint captures whether the
   co-binding arrangement is reproducible (the reaction-relevant quantity for
   two-substrate/one-product chemistry). This is the highest-risk step because
   two chemically distinct ligands coexist in one structure and must be told
   apart before matching poses across methods.

**Isolation property:** in `off` and `separate` modes, components 4 and 5 are
no-ops vs. today; multi-substrate code paths activate only for rows with >1
substrate. This keeps the blast radius small and the 197 tests green.

## Column contract

- `enzyme_id` is added in all modes (equals `Entry` for single-substrate
  rows). It is the grouping key for re-joining `separate`-mode results via
  `groupby('enzyme_id')`.
- `together`-mode analysis columns are suffixed `_s0`, `_s1`, … per substrate;
  substrate identity is recoverable from the `substrate_name` list position.
- **Single-substrate rows keep unsuffixed column names** (no `_s0`), so
  existing consumers/notebooks and the 197 tests are unaffected.
- Re-joining `separate` results needs no special step; it is a documented
  `groupby('enzyme_id')` recipe (optional summary helper may be added later).

## Error handling & edge cases

- Malformed input (list-length mismatch, empty fragment, `.` inside a name):
  fail fast at seeding with an `Entry`-named `ValueError`, before GPU work.
- `together` + one substrate: valid, behaves like `off` (no suffix, Vina not
  skipped).
- `separate` + one substrate: no expansion (Entry unchanged).
- Vina enabled + `together` + >1 substrate: Vina auto-skips per row with a log
  line; pipeline continues.
- Mixed rows (some 1-substrate, some 2-substrate in one run): each row is
  classified independently at seeding.
- Resume: expansion only when writing `_input.pkl`; the stored frame is
  already expanded/normalized, so hashes stay stable across resumes.

## Tests (TDD, failing test first per task)

Seeding / expansion (`tests/test_multi_substrate_seed.py`):
- `off`: single-substrate row unchanged.
- `separate`: `"pyr.trp"` -> 2 rows, `Entry=…__s0/__s1`, shared `enzyme_id`,
  aligned name/moiety lists.
- `separate` + single substrate: no expansion.
- `together`: row intact, lists preserved.
- Validation: list-length mismatch -> `ValueError` naming the Entry; empty
  fragment -> error.
- Resume-safe: existing `_input.pkl` not re-expanded.

Config (`tests/test_config.py` additions):
- `multi_substrate_mode` defaults to `"off"`; rejects invalid values;
  round-trips through the `init` template.

Shared helper (`tests/test_iter_substrates.py`):
- one `(smiles, name, moiety)` for single-substrate rows; N for multi.

Docking feed / skip (together):
- Boltz feed: 2-substrate row builds the expected extra ligand entity args
  (unit test of the wiring, not a live GPU run).
- Vina: together + >1 substrate -> per-row skip recorded.

Analysis steps (together): for each of the 5 steps, a 2-substrate fixture
yields `_s0`/`_s1` columns; a single-substrate fixture yields unsuffixed
columns. `ligand_rmsd` additionally emits `ligand_rmsd_joint` for a
2-substrate fixture and disambiguates the two ligands by element composition.

Regression: full suite (197 passed / 3 skipped) stays green — proves
`off`-mode backward compatibility.

## Out of scope (YAGNI)

- Live GPU verification of Chai/Boltz co-folding correctness (needs a cluster
  run); tests cover the inputs handed to the engines, not engine output.
- A dedicated "compare which substrate fits" summary step (documented
  `groupby('enzyme_id')` recipe suffices for now).
- Vina co-docking of two independent ligands (engine cannot do it).
- More than the `.`/`|` delimiter convention (no nested/quoted formats).
