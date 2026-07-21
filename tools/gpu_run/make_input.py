"""Build the all-modules GPU-run input pickle (CalB example).

Writes a one-row DataFrame with every column the enabled modules need, matching
the validated CalB inputs from tools/smoke/run_phase_c_smoke.py:
  - Candida antarctica Lipase B (CalB), UniProt P41365
  - substrate: ethyl acetate
  - catalytic triad Ser130/Asp212/His249 -> 0-indexed vina_residues 129|211|248

Output is a pickle (not CSV) so that None/list-typed columns (cofactor_moiety)
are preserved exactly. The CLI seeds from `.pkl` via pandas.read_pickle.

Usage:
  python tools/gpu_run/make_input.py --output tools/gpu_run/input.pkl
"""
import argparse
from pathlib import Path

import pandas as pd

# Full-length UniProt P41365 sequence (342 aa, includes signal peptide).
AF_SEQ = (
    "MKLLSLTGVAGVLATCVAATPLVKRLPSGSDPAFSQPKSVLDAGLTCQGASPSSVSKPIL"
    "LVPGTGTTGPQSFDSNWIPLSTQLGYTPCWISPPPFMLNDTQVNTEYMVNAITALYAGSG"
    "NNKLPVLTWSQGGLVAQWGLTFFPSIRSKVDRLMAFAPDYKGTVLAGPLDALAVSAPSVW"
    "QQTTGSALTTALRNAGGLTQIVPTTNLYSATDEIVQPQVSNSPLDSSYLFNGKNVQAQAV"
    "CGPLFVIDHAGSLTSQFSYVVGRSALRSTTGQARSADYGITDCNPLPANDLTPEQKVAAA"
    "ALLAPAAAAIVAGPKQNCEPDLMPYARPFAVGKRTCSGIVTP"
)


def build() -> pd.DataFrame:
    return pd.DataFrame({
        "Entry": ["P41365"],
        "Sequence": [AF_SEQ],
        "substrate_smiles": ["CCOC(=O)C"],
        "substrate_name": ["ethyl_acetate"],
        "substrate_moiety": ["[C](=O)([O])([O])"],
        "cofactor_smiles": [""],
        "cofactor_moiety": [None],
        # 0-indexed catalytic triad (dock_vina_step adds +1): 129|211|248.
        "vina_residues": ["129|211|248"],
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="tools/gpu_run/input.pkl")
    args = ap.parse_args()
    df = build()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    print(f"Wrote {len(df)} row(s) to {out}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
