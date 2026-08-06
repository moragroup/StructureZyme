"""Tests for the energy-aware best-pose selector in computeligandRMSD_step."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from structurezyme.steps.computeligandRMSD_step import (
    pose_energy,
    select_best_docked_structures,
)


def test_pose_energy_chai_lookup():
    scores = {"chai": {"P41365_0": -12.5}, "boltz": {}, "vina": {}}
    assert pose_energy("P41365_0_chai", scores) == -12.5


def test_pose_energy_strips_relaxed_suffix():
    scores = {"chai": {"P41365_0": -12.5}, "boltz": {}, "vina": {}}
    assert pose_energy("P41365_0_chai_relaxed", scores) == -12.5


def test_pose_energy_boltz_model_key():
    scores = {"chai": {}, "boltz": {"P41365_model_0": -9.0}, "vina": {}}
    assert pose_energy("P41365_model_0_boltz_relaxed", scores) == -9.0


def test_pose_energy_vina_integer_key():
    scores = {"chai": {}, "boltz": {}, "vina": {3: -7.2}}
    assert pose_energy("P41365_3_vina_relaxed", scores) == -7.2


def test_pose_energy_missing_returns_none():
    scores = {"chai": {"P41365_0": -12.5}, "boltz": {}, "vina": {}}
    assert pose_energy("P41365_9_chai", scores) is None


def test_pose_energy_none_scores_returns_none():
    assert pose_energy("P41365_0_chai", None) is None
    assert pose_energy("P41365_0_chai", {}) is None
