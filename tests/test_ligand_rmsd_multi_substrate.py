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
