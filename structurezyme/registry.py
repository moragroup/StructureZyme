# structurezyme/registry.py
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class StepSpec:
    name: str
    depends_on: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    runner: Callable | None = None

    @property
    def output(self) -> str:
        return self.name

def _spec(name, depends_on=None, inputs=None):
    return StepSpec(name=name, depends_on=depends_on or [], inputs=inputs or [])

STEPS: dict[str, StepSpec] = {
    "squidly":          _spec("squidly"),
    "chai":             _spec("chai", ["squidly"], ["squidly"]),
    "boltz":            _spec("boltz", ["chai"], ["chai"]),
    "vina":             _spec("vina", ["boltz"], ["boltz"]),
    "docking_metrics":  _spec("docking_metrics", ["boltz"], ["boltz", "vina"]),
    "prepare_files":    _spec("prepare_files", ["docking_metrics"], ["docking_metrics"]),
    "fastrelax":        _spec("fastrelax", ["prepare_files"], ["prepare_files"]),
    # fastrelax is optional but, when enabled, must run *before* superimpose
    # (superimpose consumes its relaxed frame). Declaring it in depends_on
    # enforces the ordering; a disabled fastrelax is simply skipped at run time.
    "superimpose":      _spec("superimpose", ["prepare_files", "fastrelax"], ["prepare_files", "fastrelax"]),
    "protein_rmsd":     _spec("protein_rmsd", ["superimpose"], ["superimpose"]),
    "ligand_rmsd":      _spec("ligand_rmsd", ["protein_rmsd"], ["protein_rmsd"]),
    "geometric_filter": _spec("geometric_filter", ["ligand_rmsd"], ["ligand_rmsd"]),
    "fpocket":          _spec("fpocket", ["geometric_filter"], ["geometric_filter"]),
    "ligand_sasa":      _spec("ligand_sasa", ["fpocket"], ["fpocket"]),
    "plip":             _spec("plip", ["ligand_sasa"], ["ligand_sasa"]),
    "placer":           _spec("placer", ["plip"], ["plip"]),
}

def ordered_steps() -> list[str]:
    visited, temp, order = set(), set(), []
    def visit(n):
        if n in visited:
            return
        if n in temp:
            raise ValueError(f"Cycle detected at {n}")
        temp.add(n)
        for dep in STEPS[n].depends_on:
            visit(dep)
        temp.discard(n)
        visited.add(n)
        order.append(n)
    for name in STEPS:
        visit(name)
    return order


from . import step_runners as _sr  # noqa: E402

_RUNNERS = {
    "squidly": _sr.run_squidly, "chai": _sr.run_chai, "boltz": _sr.run_boltz,
    "vina": _sr.run_vina, "docking_metrics": _sr.run_docking_metrics,
    "prepare_files": _sr.run_prepare_files, "fastrelax": _sr.run_fastrelax,
    "superimpose": _sr.run_superimpose, "protein_rmsd": _sr.run_protein_rmsd,
    "ligand_rmsd": _sr.run_ligand_rmsd, "geometric_filter": _sr.run_geometric_filter,
    "fpocket": _sr.run_fpocket, "ligand_sasa": _sr.run_ligand_sasa,
    "plip": _sr.run_plip, "placer": _sr.run_placer,
}
for _name, _fn in _RUNNERS.items():
    STEPS[_name].runner = _fn
