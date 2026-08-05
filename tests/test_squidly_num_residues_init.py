"""Squidly.__init__ accepts and validates num_residues."""
from __future__ import annotations

import pytest

from structurezyme.steps.squidly_step import Squidly


def test_default_num_residues_is_none():
    assert Squidly().num_residues is None


def test_positive_int_is_stored():
    assert Squidly(num_residues=3).num_residues == 3


@pytest.mark.parametrize("bad", [0, -1, 2.5, "3"])
def test_invalid_num_residues_raises(bad):
    with pytest.raises(ValueError):
        Squidly(num_residues=bad)
