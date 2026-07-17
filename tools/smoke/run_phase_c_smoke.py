"""Phase C smoke test: full pipeline with Squidly catalytic-residue prediction.

Run this on a GPU node with the structurezyme env activated and a real Boltz
cache directory. It exercises the full chain:
  Squidly -> Chai -> Boltz -> Vina -> DockingMetrics

Prerequisites:
  - conda activate structurezyme
  - `squidly` on $PATH (verify with `which squidly`)
  - a Boltz cache directory (run `boltz predict example.yml --cache <dir>`)
  - a GPU (ESM2 3B inference needs CUDA)
  - Vina toolchain: vina CLI, obabel, meeko (mk_prepare_ligand.py),
    vina Python module -- all installed in the structurezyme env.

Usage:
  python tools/smoke/run_phase_c_smoke.py --boltz-cache /path/to/boltz/cache \
                              --output-dir pipeline_output_phaseC
"""
import argparse
from pathlib import Path

import pandas as pd

from structurezyme.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boltz-cache", required=True,
        help="Path to the Boltz cache directory (required)"
    )
    parser.add_argument(
        "--output-dir", default="pipeline_output_phaseC",
        help="Base output directory (default: pipeline_output_phaseC)"
    )
    args = parser.parse_args()

    # Candida antarctica Lipase B (CalB), UniProt P41365.
    # - Real serine hydrolase, canonical alpha/beta hydrolase fold.
    # - Full-length UniProt sequence (342 aa, includes 25-aa signal peptide).
    # - AlphaFold DB has AF-P41365-F1 (pLDDT 94.94) - this is what the Vina
    #   step downloads automatically via get_alphafold_structure(Entry, ...).
    # - Catalytic triad in the FULL-LENGTH sequence: Ser130, Asp212, His249
    #   (verified: positions land on S/D/H with TWSQG nucleophile elbow).
    # - Standard substrate: ethyl acetate.
    #
    # NOTE on vina_residues indexing: dock_vina_step.py:36 does
    # `int(r) + 1` on each pipe-separated value, so the values stored here
    # must be 0-indexed. Triad 1-indexed (130, 212, 249) -> 0-indexed
    # (129, 211, 248).
    af_seq = (
        "MKLLSLTGVAGVLATCVAATPLVKRLPSGSDPAFSQPKSVLDAGLTCQGASPSSVSKPIL"
        "LVPGTGTTGPQSFDSNWIPLSTQLGYTPCWISPPPFMLNDTQVNTEYMVNAITALYAGSG"
        "NNKLPVLTWSQGGLVAQWGLTFFPSIRSKVDRLMAFAPDYKGTVLAGPLDALAVSAPSVW"
        "QQTTGSALTTALRNAGGLTQIVPTTNLYSATDEIVQPQVSNSPLDSSYLFNGKNVQAQAV"
        "CGPLFVIDHAGSLTSQFSYVVGRSALRSTTGQARSADYGITDCNPLPANDLTPEQKVAAA"
        "ALLAPAAAAIVAGPKQNCEPDLMPYARPFAVGKRTCSGIVTP"
    )
    df = pd.DataFrame({
        'Entry': ['P41365'],  # UniProt accession -> triggers AF2 auto-download
        'Sequence': [af_seq],
        'substrate_smiles': ['CCOC(=O)C'],
        'substrate_name': ['ethyl_acetate'],
        'substrate_moiety': ['[C](=O)([O])([O])'],
        # CalB ester hydrolysis has no small-molecule cofactor; Chai's
        # run_chai (docko/chai.py:51) skips the cofactor block when
        # cofactor_smiles == "". Empty list/string also avoids the
        # 'NoneType is not iterable' bug seen previously. The
        # GeneralGeometricFiltering step at
        # geometric_filtering_cofactor_MCS.py:452-454 reads
        # row['cofactor_moiety'] unconditionally, so the column must exist.
        'cofactor_smiles': [""],
        'cofactor_moiety': [None],
        # Catalytic triad of CalB (0-indexed because dock_vina_step.py:36
        # adds +1): Ser130, Asp212, His249 -> 129|211|248.
        'vina_residues': ['129|211|248'],
    })

    # Vina toolchain (installed in structurezyme env earlier in this session):
    #   - obabel               : pre-existing
    #   - mk_prepare_ligand.py : pip install meeko
    #   - vina CLI v1.2.5      : static binary in envs/structurezyme/bin/vina
    #   - vina Python lib      : pip install vina
    # docko/helpers.py patched to invoke mk_prepare_ligand.py directly
    # instead of `conda run -n vina ...`.
    #
    # Vina obtains its 3D structure via dock_vina_step.py:42-51 by
    # downloading from AlphaFold DB using Entry as the UniProt accession.
    # P41365 has an AF2 model with pLDDT 94.94.
    pipeline = Pipeline(
        df=df,
        boltz_cache_dir=args.boltz_cache,
        base_output_dir=args.output_dir,
        run_vina=True,
    )
    pipeline.run()

    # Verify Squidly output
    squidly_pkl = Path(args.output_dir) / "docking" / "squidly.pkl"
    if squidly_pkl.exists():
        df_out = pd.read_pickle(squidly_pkl)
        print("\n=== Squidly output ===")
        print(f"Rows: {len(df_out)}")
        if 'catalytic_residues' in df_out.columns:
            print(f"catalytic_residues: {df_out['catalytic_residues'].tolist()}")
        if 'Squidly_CR_Position' in df_out.columns:
            print(f"Squidly_CR_Position: {df_out['Squidly_CR_Position'].tolist()}")
    else:
        print(f"WARNING: {squidly_pkl} not found")


if __name__ == "__main__":
    main()
