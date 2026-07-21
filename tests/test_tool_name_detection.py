"""Tool-name detection must survive the fastrelax `_relaxed` suffix.

fastrelax renames poses `<stem>.pdb -> <stem>_relaxed.pdb`
(structurezyme/steps/fastrelax_step.py:266). The RMSD steps classify each
pose by docking tool via `get_tool_from_structure_name`, which takes the last
`_`-delimited token. Without stripping `_relaxed` that token is "relaxed" for
every relaxed pose, collapsing chai/boltz/vina into one fake tool -- which
crashed ligand_rmsd and silently corrupted protein_rmsd's `tool` column.
"""
import pytest

from structurezyme.steps.computeligandRMSD_step import (
    get_tool_from_structure_name as ligand_tool,
)
from structurezyme.steps.computeproteinRMSD_step import (
    get_tool_from_structure_name as protein_tool,
)


@pytest.mark.parametrize("fn", [ligand_tool, protein_tool])
@pytest.mark.parametrize(
    "name,expected",
    [
        # plain (unrelaxed) pose names
        ("Q97WW0_1_vina", "vina"),
        ("P41365_0_chai", "chai"),
        ("P41365_model_0_boltz", "boltz"),
        # relaxed pose names (the regression): tool must still be detected
        ("P41365_0_chai_relaxed", "chai"),
        ("P41365_4_chai_relaxed", "chai"),
        ("P41365_model_0_boltz_relaxed", "boltz"),
        ("Q97WW0_1_vina_relaxed", "vina"),
    ],
)
def test_tool_detected_through_relaxed_suffix(fn, name, expected):
    assert fn(name) == expected


@pytest.mark.parametrize("fn", [ligand_tool, protein_tool])
def test_no_underscore_falls_back(fn):
    assert fn("noseparator") == "UNKNOWN_tool"
