"""Build the multi-row all-modules GPU-run input pickle (3 enzymes).

Multi-row validation batch for the resumable pipeline. Three heterogeneous
rows exercise per-row iteration, N-vs-N RMSD matrices, tool detection, the
per-row cofactor conditional in geometric_filter, and the fail-loud missing-PDB
guards -- all with num_threads=1 (see run_multirow.yml for why >1 is unsafe
in the external enzymetk docking steps).

Rows:
  1. CalB (P41365)      - Candida antarctica Lipase B, esterase, NO cofactor.
  2. F5SYD3             - flavin-dependent monooxygenase, indole, FAD cofactor.
  3. A0A410H4M7         - flavin-dependent monooxygenase, indole, FAD cofactor.

Design choices for this validation run:
  - vina_residues left EMPTY for all rows -> squidly predicts catalytic
    residues (as_threshold unset -> squidly self-calibrates), then vina docks
    at squidly's residues. Tests the full auto squidly->vina path.
  - substrate_moiety / cofactor_moiety left empty -> geometric_filter falls
    back to whole-ligand centroid (plumbing validation, not exact geometry).

Usage:
  python tools/gpu_run/make_input_multirow.py --output tools/gpu_run/input_multirow.pkl
"""
import argparse
from pathlib import Path

import pandas as pd

# Full-length UniProt P41365 (CalB, 342 aa).
CALB_SEQ = (
    "MKLLSLTGVAGVLATCVAATPLVKRLPSGSDPAFSQPKSVLDAGLTCQGASPSSVSKPIL"
    "LVPGTGTTGPQSFDSNWIPLSTQLGYTPCWISPPPFMLNDTQVNTEYMVNAITALYAGSG"
    "NNKLPVLTWSQGGLVAQWGLTFFPSIRSKVDRLMAFAPDYKGTVLAGPLDALAVSAPSVW"
    "QQTTGSALTTALRNAGGLTQIVPTTNLYSATDEIVQPQVSNSPLDSSYLFNGKNVQAQAV"
    "CGPLFVIDHAGSLTSQFSYVVGRSALRSTTGQARSADYGITDCNPLPANDLTPEQKVAAA"
    "ALLAPAAAAIVAGPKQNCEPDLMPYARPFAVGKRTCSGIVTP"
)

# Flavin-dependent monooxygenase F5SYD3 (456 aa).
F5SYD3_SEQ = (
    "MATRIAILGAGPSGMAQLRAFQSAQEKGAEIPELVCFEKQADWGGQWNYTWRTGLDENGEP"
    "VHSSMYRYLWSNGPKECLEFADYTFDEHFGKPIASYPPREVLWDYIKGRVEKAGVRKYIRF"
    "NTAVRHVEFNEDSQTFTVTVQDHTTDTIYSEEFDYVVCCTGHFSTPYVPEFEGFEKFGGRI"
    "LHAHDFRDALEFKDKTVLLVGSSYSAEDIGSQCYKYGAKKLISCYRTAPMGYKWPENWDER"
    "PNLVRVDTENAYFADGSSEKVDAIILCTGYIHHFPFLNDDLRLVTNNRLWPLNLYKGVVWE"
    "DNPKFFYIGMQDQWYSFNMFDAQAWYARDVIMGRLPLPSKEEMKADSMAWREKELTLVTAE"
    "EMYTYQGDYIQNLIDMTDYPSFDIPATNKTFLEWKHHKKENIMTFRDHSYRSLMTGTMAPK"
    "HHTPWIDALDDSLEAYLSDKSEIPVAKEA"
)

# Flavin-dependent monooxygenase A0A410H4M7 (457 aa).
A0A410H4M7_SEQ = (
    "MTKRVAIIGAGPSGLAQLRAFQSAQSKGADIPELVCFEKQSDWGGLWNYTWRTGLDENGEP"
    "VHCSMYRYLWSNGPKECLEFADYTFDEHFGKPIASYPPREVLWDYIKGRVEKAGVRDYIRF"
    "NTVVRNVSYDDASETFTVTVQDHNEDKIYSEEFDYVISASGHFSTPKVPEFEGFQTFGGRI"
    "LHAHDFRDALEFKDKDILLVGASYSAEDIGSQCYKYGAKSITTCFRSAPMGYKWPENWEER"
    "PLLERVDTHRAYFADGSSKKIDAIILCTGYLHHFPYLPDSLRLVTDNRLWPLDLYKGVVWE"
    "DNPKFFYLGMQDQWYTFNMFDAQAWYVRDIILGRIPLPSKAEMTQNSQAWREKELKLETAE"
    "EMYTFQGDYIQELIDATDYPSFDIPAVNQTFLEWKHHKKENIMTFRDHSYRSLMTGTMSPK"
    "HHTPWIDALDDSLEAYLADGPQDEKLASNQ"
)

# Indole substrate; FAD cofactor (for the two FMO rows).
INDOLE_SMILES = "C1=CC=C2C(=C1)C=CN2"
FAD_SMILES = (
    "CC1=CC2=C(C=C1C)N(C3=NC(=O)NC(=O)C3=N2)C[C@@H]([C@@H]([C@@H]"
    "(COP(=O)(O)OP(=O)(O)OC[C@@H]4[C@H]([C@H]([C@@H](O4)N5C=NC6=C"
    "(N=CN=C65)N)O)O)O)O)O"
)


def build() -> pd.DataFrame:
    return pd.DataFrame({
        "Entry": ["P41365", "F5SYD3", "A0A410H4M7"],
        "Sequence": [CALB_SEQ, F5SYD3_SEQ, A0A410H4M7_SEQ],
        "substrate_smiles": ["CCOC(=O)C", INDOLE_SMILES, INDOLE_SMILES],
        "substrate_name": ["ethyl_acetate", "indole", "indole"],
        # Empty moiety -> geometric_filter uses whole-ligand centroid.
        "substrate_moiety": ["", "", ""],
        # CalB has no cofactor; both FMOs use FAD.
        "cofactor_smiles": ["", FAD_SMILES, FAD_SMILES],
        "cofactor_moiety": [None, None, None],
        # Empty -> squidly predicts catalytic residues, then vina docks there.
        "vina_residues": ["", "", ""],
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="tools/gpu_run/input_multirow.pkl")
    args = ap.parse_args()
    df = build()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    print(f"Wrote {len(df)} row(s) to {out}")
    print("Columns:", list(df.columns))
    print(df[["Entry", "substrate_name", "cofactor_smiles"]].to_string(index=False))


if __name__ == "__main__":
    main()
