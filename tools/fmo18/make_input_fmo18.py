"""Build the 18-row FMO input pickle (indole substrate, FAD cofactor).

All 18 rows are flavin-dependent monooxygenase (FMO) family enzymes. Substrate
is indole; cofactor is FAD (the prosthetic flavin that forms the C4a-hydro-
peroxide intermediate directly attacking the substrate).

Usage:
  python tools/fmo18/make_input_fmo18.py --output tools/fmo18/input_fmo18.pkl
"""
import argparse
from pathlib import Path

import pandas as pd

from structurezyme_fmo18_sequences import SEQUENCES

INDOLE_SMILES = "C1=CC=C2C(=C1)C=CN2"
FAD_SMILES = (
    "CC1=CC2=C(C=C1C)N(C3=NC(=O)NC(=O)C3=N2)C[C@@H]([C@@H]([C@@H]"
    "(COP(=O)(O)OP(=O)(O)OC[C@@H]4[C@H]([C@H]([C@@H](O4)N5C=NC6=C"
    "(N=CN=C65)N)O)O)O)O)O"
)


def build() -> pd.DataFrame:
    entries = [entry for entry, _seq in SEQUENCES]
    seqs = [seq for _entry, seq in SEQUENCES]
    n = len(entries)
    return pd.DataFrame({
        "Entry": entries,
        "Sequence": seqs,
        "substrate_smiles": [INDOLE_SMILES] * n,
        "substrate_name": ["indole"] * n,
        "substrate_moiety": [""] * n,
        "cofactor_smiles": [FAD_SMILES] * n,
        "cofactor_moiety": [None] * n,
        "vina_residues": [""] * n,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="tools/fmo18/input_fmo18.pkl")
    args = ap.parse_args()
    df = build()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    print(f"Wrote {len(df)} row(s) to {out}")
    print("Columns:", list(df.columns))
    print(df[["Entry", "substrate_name"]].to_string(index=False))


if __name__ == "__main__":
    main()
