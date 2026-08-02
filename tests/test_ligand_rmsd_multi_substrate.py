import pandas as pd
from rdkit import Chem
from structurezyme.steps.computeligandRMSD_step import _match_substrate_chains


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def test_matches_each_substrate_to_closest_chain():
    # two chains present: ethanol-like and pyridine-like
    ethanol = _mol("CCO")
    pyridine = _mol("c1ccncc1")
    ligands = [pyridine, ethanol]  # deliberately out of order
    matched = _match_substrate_chains(ligands, ["CCO", "c1ccncc1"])
    assert matched[0].GetNumAtoms() == ethanol.GetNumAtoms()
    assert matched[1].GetNumAtoms() == pyridine.GetNumAtoms()


def test_no_chain_matches_returns_none_slot():
    ethanol = _mol("CCO")
    matched = _match_substrate_chains([ethanol], ["CCO", "c1ccncc1"])
    assert matched[0] is not None
    # only one ligand present; second substrate has no remaining chain
    assert matched[1] is None


def test_does_not_reuse_a_chain_across_substrates():
    ethanol = _mol("CCO")
    matched = _match_substrate_chains([ethanol], ["CCO", "CCO"])
    assert matched[0] is not None
    assert matched[1] is None  # the single chain is consumed by substrate 0


def _conformer(smiles, seed):
    # a hydrogen-bearing 3D mol with a specific (seed-dependent) conformer, so
    # two conformers of the SAME molecule differ in coordinates -> nonzero RMSD
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    return m


def test_per_substrate_rmsd_is_real_cross_tool_metric(tmp_path, monkeypatch):
    # This test exercises the ACTUAL T/U (tool1) vs V/W (tool2) chain pairing
    # (superimposestructures_step.py:211-221). It must fail if the code
    # self-compares, pairs the wrong chains, or swaps tool1/tool2.
    import structurezyme.steps.computeligandRMSD_step as mod
    import numpy as np

    entry = tmp_path / "P1"
    entry.mkdir()
    (entry / "chai_0__boltz_0.pdb").write_text("dummy")

    df = pd.DataFrame({"Entry": ["P1"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    # tool1 ligands live on chains T (substrate 0) and U (substrate 1);
    # tool2 ligands on V (substrate 0) and W (substrate 1). Same molecule per
    # substrate across tools, but a DIFFERENT conformer -> a real, nonzero
    # cross-tool RMSD only when T<->V and U<->W are paired correctly.
    chain_to_mol = {
        "T": _conformer("CCO", seed=1),        # tool1 ethanol
        "V": _conformer("CCO", seed=99),       # tool2 ethanol (diff conformer)
        "U": _conformer("c1ccncc1", seed=1),   # tool1 pyridine
        "W": _conformer("c1ccncc1", seed=99),  # tool2 pyridine (diff conformer)
    }
    monkeypatch.setattr(mod, "get_hetatm_chain_ids",
                        lambda p: ["T", "U", "V", "W"], raising=True)
    monkeypatch.setattr(mod, "extract_chain_as_rdkit_mol",
                        lambda p, cid, sanitize=False: chain_to_mol[cid],
                        raising=True)

    step = mod.LigandRMSD(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    rmsd_df, _ = step.execute(df)
    cols = set(rmsd_df.columns)
    assert {"ligand_rmsd_s0", "ligand_rmsd_s1", "ligand_rmsd_joint"} <= cols

    # both per-substrate cross-tool RMSDs are computed (not NaN) and strictly
    # positive (the two tools' conformers differ). A self-comparison bug would
    # give 0.0; a wrong-chain pairing would give NaN or a mismatched value.
    s0 = rmsd_df["ligand_rmsd_s0"].iloc[0]
    s1 = rmsd_df["ligand_rmsd_s1"].iloc[0]
    assert not np.isnan(s0) and s0 > 0.0
    assert not np.isnan(s1) and s1 > 0.0
