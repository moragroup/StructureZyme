"""Unit tests for _select_top_n_from_ensemble (pure top-N by mean)."""
from __future__ import annotations

import numpy as np

from structurezyme.steps.squidly_step import _select_top_n_from_ensemble


def test_picks_n_highest_mean_ordered_by_index():
    mean = [0.01, 0.90, 0.02, 0.80, 0.70]
    # top-3 by value are indices 1(0.90), 3(0.80), 4(0.70); output ascending index.
    assert _select_top_n_from_ensemble(mean, 3) == "1|3|4"


def test_n_larger_than_length_returns_all():
    mean = [0.2, 0.5, 0.1]
    assert _select_top_n_from_ensemble(mean, 10) == "0|1|2"


def test_ties_broken_by_lower_index():
    mean = [0.5, 0.5, 0.5, 0.1]
    # three-way tie at 0.5; pick the two lowest indices.
    assert _select_top_n_from_ensemble(mean, 2) == "0|1"


def test_accepts_numpy_array():
    mean = np.array([0.1, 0.9, 0.4])
    assert _select_top_n_from_ensemble(mean, 1) == "1"


def test_empty_or_none_returns_empty_string():
    assert _select_top_n_from_ensemble(None, 3) == ""
    assert _select_top_n_from_ensemble([], 3) == ""
