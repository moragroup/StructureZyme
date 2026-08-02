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


def test_per_substrate_rmsd_keys_emitted(tmp_path, monkeypatch):
    import structurezyme.steps.computeligandRMSD_step as mod

    # one entry dir with one paired PDB
    entry = tmp_path / "P1"
    entry.mkdir()
    (entry / "chai_0__boltz_0.pdb").write_text("dummy")

    df = pd.DataFrame({"Entry": ["P1"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    # stub chain extraction: 2 chains per structure, matching the 2 substrates
    from rdkit import Chem
    eth = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    pyr = Chem.AddHs(Chem.MolFromSmiles("c1ccncc1"))
    Chem.AllChem.EmbedMolecule(eth); Chem.AllChem.EmbedMolecule(pyr)
    monkeypatch.setattr(mod, "get_hetatm_chain_ids",
                        lambda p: ["B", "C"], raising=True)
    monkeypatch.setattr(mod, "extract_chain_as_rdkit_mol",
                        lambda p, cid, sanitize=False:
                            (eth if cid == "B" else pyr), raising=True)

    step = mod.LigandRMSD(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    rmsd_df, _ = step.execute(df)
    cols = set(rmsd_df.columns)
    assert "ligand_rmsd_s0" in cols
    assert "ligand_rmsd_s1" in cols
    assert "ligand_rmsd_joint" in cols
