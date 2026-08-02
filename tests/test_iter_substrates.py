# tests/test_iter_substrates.py
import pandas as pd
from structurezyme.utils.helpers import iter_substrates


def test_single_substrate_yields_one_tuple():
    row = {"substrate_smiles": "CCO", "substrate_name": "ethanol",
           "substrate_moiety": "CO"}
    assert iter_substrates(row) == [("CCO", "ethanol", "CO")]


def test_two_substrates_aligned():
    row = {"substrate_smiles": "CCO.c1ccncc1",
           "substrate_name": "ethanol|pyridine",
           "substrate_moiety": "CO|n1ccccc1"}
    assert iter_substrates(row) == [
        ("CCO", "ethanol", "CO"),
        ("c1ccncc1", "pyridine", "n1ccccc1"),
    ]


def test_missing_name_and_moiety_columns_pad_empty():
    row = {"substrate_smiles": "CCO.c1ccncc1"}
    assert iter_substrates(row) == [("CCO", "", ""), ("c1ccncc1", "", "")]


def test_short_parallel_list_pads_empty():
    row = {"substrate_smiles": "CCO.c1ccncc1",
           "substrate_name": "ethanol"}
    assert iter_substrates(row) == [("CCO", "ethanol", ""), ("c1ccncc1", "", "")]


def test_accepts_pandas_series():
    s = pd.Series({"substrate_smiles": "CCO.O", "substrate_name": "a|b",
                   "substrate_moiety": "x|y"})
    assert iter_substrates(s) == [("CCO", "a", "x"), ("O", "b", "y")]


def test_nan_optional_columns_treated_as_absent():
    s = pd.Series({"substrate_smiles": "CCO.O",
                   "substrate_name": float("nan"),
                   "substrate_moiety": None})
    assert iter_substrates(s) == [("CCO", "", ""), ("O", "", "")]
