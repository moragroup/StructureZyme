# tests/test_expand_substrates.py
import pandas as pd
import pytest
from structurezyme.runner import _expand_substrates


def _frame():
    return pd.DataFrame({
        "Entry": ["P1", "P2"],
        "Sequence": ["M", "K"],
        "substrate_smiles": ["CCO.c1ccncc1", "O"],
        "substrate_name": ["ethanol|pyridine", "water"],
        "substrate_moiety": ["CO|n1ccccc1", "O"],
    })


def test_off_mode_adds_enzyme_id_only():
    df = _expand_substrates(_frame(), "off")
    assert list(df["Entry"]) == ["P1", "P2"]
    assert list(df["enzyme_id"]) == ["P1", "P2"]
    assert list(df["substrate_smiles"]) == ["CCO.c1ccncc1", "O"]


def test_together_mode_adds_enzyme_id_only():
    df = _expand_substrates(_frame(), "together")
    assert list(df["Entry"]) == ["P1", "P2"]
    assert list(df["enzyme_id"]) == ["P1", "P2"]
    assert list(df["substrate_smiles"]) == ["CCO.c1ccncc1", "O"]


def test_separate_mode_expands_multi_keeps_single():
    df = _expand_substrates(_frame(), "separate").reset_index(drop=True)
    assert list(df["Entry"]) == ["P1__s0", "P1__s1", "P2"]
    assert list(df["enzyme_id"]) == ["P1", "P1", "P2"]
    assert list(df["substrate_smiles"]) == ["CCO", "c1ccncc1", "O"]
    assert list(df["substrate_name"]) == ["ethanol", "pyridine", "water"]
    assert list(df["substrate_moiety"]) == ["CO", "n1ccccc1", "O"]
    # non-substrate columns are carried through per expanded row
    assert list(df["Sequence"]) == ["M", "M", "K"]


def test_separate_mode_single_substrate_not_suffixed():
    one = pd.DataFrame({"Entry": ["P9"], "Sequence": ["M"],
                        "substrate_smiles": ["O"], "substrate_name": ["water"],
                        "substrate_moiety": ["O"]})
    df = _expand_substrates(one, "separate")
    assert list(df["Entry"]) == ["P9"]
    assert list(df["enzyme_id"]) == ["P9"]


def test_separate_length_mismatch_raises():
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.O"],
                       "substrate_name": ["only_one"],  # 1 name for 2 smiles
                       "substrate_moiety": ["CO|O"]})
    with pytest.raises(ValueError, match="P1"):
        _expand_substrates(df, "separate")


def test_empty_fragment_raises():
    df = pd.DataFrame({"Entry": ["P2"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO..O"]})  # empty middle fragment
    with pytest.raises(ValueError, match="P2"):
        _expand_substrates(df, "together")


def test_off_mode_does_not_validate():
    # off mode must never raise even on ragged parallel lists
    df = pd.DataFrame({"Entry": ["P3"], "substrate_smiles": ["CCO.O"],
                       "substrate_name": ["only_one"]})
    _expand_substrates(df, "off")  # no exception
