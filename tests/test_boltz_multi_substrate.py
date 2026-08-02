# tests/test_boltz_multi_substrate.py
import pandas as pd
import pytest
from structurezyme.step_runners import _boltz_together_frame


def test_two_substrates_no_cofactor_routes_second_to_cofactor():
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})
    out = _boltz_together_frame(df)
    assert list(out["substrate_smiles"]) == ["CCO"]
    assert list(out["cofactor_smiles"]) == ["c1ccncc1"]


def test_single_substrate_passthrough_empty_cofactor():
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO"]})
    out = _boltz_together_frame(df)
    assert list(out["substrate_smiles"]) == ["CCO"]
    assert list(out["cofactor_smiles"]) == [""]


def test_three_substrates_rejected():
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.O.N"]})
    with pytest.raises(ValueError, match="at most 2 substrates"):
        _boltz_together_frame(df)


def test_two_substrates_with_existing_cofactor_rejected():
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.c1ccncc1"],
                       "cofactor_smiles": ["[Fe]"]})
    with pytest.raises(ValueError, match="pre-existing cofactor"):
        _boltz_together_frame(df)
