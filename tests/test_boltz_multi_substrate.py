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


def test_two_substrates_with_empty_or_nan_cofactor_treated_as_absent():
    # an empty-string or NaN cofactor cell is NOT a real cofactor: the 2nd
    # substrate is still routed into the cofactor slot (no rejection)
    df = pd.DataFrame({"Entry": ["P1", "P2"], "Sequence": ["M", "M"],
                       "substrate_smiles": ["CCO.c1ccncc1", "CCO.c1ccncc1"],
                       "cofactor_smiles": ["", float("nan")]})
    out = _boltz_together_frame(df)
    assert list(out["substrate_smiles"]) == ["CCO", "CCO"]
    assert list(out["cofactor_smiles"]) == ["c1ccncc1", "c1ccncc1"]


def test_multi_row_frame_routes_each_row_positionally():
    # non-default index must not break the positional iloc[pos] cofactor lookup;
    # each row is routed independently (2-substrate routed, single passed through)
    df = pd.DataFrame(
        {"Entry": ["P1", "P2"], "Sequence": ["M", "M"],
         "substrate_smiles": ["CCO.c1ccncc1", "CCO"],
         "cofactor_smiles": ["", "[Fe]"]},
        index=[7, 3],
    )
    out = _boltz_together_frame(df)
    assert list(out["substrate_smiles"]) == ["CCO", "CCO"]
    # row 0: 2nd substrate -> cofactor; row 1: single substrate keeps its prior cofactor
    assert list(out["cofactor_smiles"]) == ["c1ccncc1", "[Fe]"]
