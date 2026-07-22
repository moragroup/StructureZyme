# Vendored: Rosetta molfile_to_params.py + rosetta_py

## Provenance

Fetched from the RosettaCommons/rosetta monorepo, `main` branch, on 2026-07-22:

```
https://raw.githubusercontent.com/RosettaCommons/rosetta/main/source/scripts/python/public/
  molfile_to_params.py
  rosetta_py/__init__.py
  rosetta_py/io/__init__.py
  rosetta_py/io/mdl_molfile.py
  rosetta_py/utility/__init__.py
  rosetta_py/utility/rankorder.py
  rosetta_py/utility/r3.py
```

## Why vendored

`molfile_to_params.py` is required by `structurezyme/steps/fastrelax_step.py` to
generate PyRosetta `.params` files for non-canonical ligands (indole, FAD, etc.).
Without it, `pose_from_pdb` silently maps every `LIG` residue to a hardcoded
29-atom generic template, mangling the chemistry of both substrates and
cofactors. See `docs/known-issues.md` for details.

The script is NOT shipped with `pyrosetta` (any recent wheel) and is NOT in
`RosettaCommons/tools`. It only lives in the full Rosetta monorepo, which
we do not want to require as a runtime dependency.

Vendoring keeps the pipeline reproducible and offline-friendly.

## License

Rosetta is distributed under a modified BSD license permitting redistribution
for academic and commercial use. See the header of `molfile_to_params.py` and
https://www.rosettacommons.org/software/license-and-download for details.

## How to update

To pull a newer upstream copy:

```bash
BASE="https://raw.githubusercontent.com/RosettaCommons/rosetta/main/source/scripts/python/public"
curl -sL -o molfile_to_params.py         "$BASE/molfile_to_params.py"
curl -sL -o rosetta_py/__init__.py       "$BASE/rosetta_py/__init__.py"
curl -sL -o rosetta_py/io/__init__.py    "$BASE/rosetta_py/io/__init__.py"
curl -sL -o rosetta_py/io/mdl_molfile.py "$BASE/rosetta_py/io/mdl_molfile.py"
curl -sL -o rosetta_py/utility/__init__.py "$BASE/rosetta_py/utility/__init__.py"
curl -sL -o rosetta_py/utility/rankorder.py "$BASE/rosetta_py/utility/rankorder.py"
curl -sL -o rosetta_py/utility/r3.py     "$BASE/rosetta_py/utility/r3.py"
```

Then run the regression suite in `tests/` to verify compatibility.

## Usage

The script is invoked as a subprocess from `structurezyme/utils/ligand_params.py`.
Do NOT `import` these modules directly from application code — they use
`print()` and `optparse` and are designed to run as a standalone script.
