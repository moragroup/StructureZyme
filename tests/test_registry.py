# tests/test_registry.py
from structurezyme.registry import STEPS, ordered_steps

def test_all_steps_present():
    expected = {"squidly","chai","boltz","vina","docking_metrics","prepare_files",
                "fastrelax","superimpose","protein_rmsd","ligand_rmsd",
                "geometric_filter","fpocket","ligand_sasa","plip","placer"}
    assert set(STEPS) == expected

def test_topological_order_respects_dependencies():
    order = ordered_steps()
    idx = {name: i for i, name in enumerate(order)}
    for name, spec in STEPS.items():
        for dep in spec.depends_on:
            assert idx[dep] < idx[name], f"{dep} must precede {name}"
