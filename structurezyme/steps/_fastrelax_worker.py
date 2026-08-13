"""Standalone PyRosetta FastRelax worker.

Run under the RosettaFastRelax venv python (py3.12), NOT the structurezyme
env. Imports only stdlib + pyrosetta so it never touches structurezyme code.

Usage:
    python _fastrelax_worker.py <spec.json>

Reads a JSON spec (see keys below), runs FastRelax, writes the relaxed PDB to
`out_pdb`, and prints `{"relaxed_path": ..., "score": ...}` as the LAST line of
stdout. On any error prints `{"error": ...}` and exits 1.
"""
import json
import sys


def _run(spec: dict) -> dict:
    import pyrosetta

    extra_res_fa = " ".join(spec["extra_res_fa"])
    init_opts = "-mute all"
    if extra_res_fa:
        init_opts = f"{init_opts} -extra_res_fa {extra_res_fa}"
    pyrosetta.init(extra_options=init_opts, silent=True)

    pose = pyrosetta.pose_from_pdb(spec["prepared_pdb"])
    mode = spec["mode"]
    ligand_resname = spec.get("ligand_resname")
    shell_radius = spec["shell_radius"]
    constraint_weight = spec["constraint_weight"]
    scorefunction = spec["scorefunction"]

    movemap = None
    movemap_factory = None
    if mode == "ligand_focused":
        from pyrosetta.rosetta.core.select.residue_selector import (
            NeighborhoodResidueSelector,
            ResidueNameSelector,
        )
        from pyrosetta.rosetta.core.select.movemap import (
            MoveMapFactory,
            move_map_action,
        )
        from pyrosetta.rosetta.protocols.constraint_generator import (
            AddConstraints,
            CoordinateConstraintGenerator,
        )

        if ligand_resname is None:
            raise RuntimeError(
                "mode='ligand_focused' requires a ligand_resname; got None."
            )
        ligand_sel = ResidueNameSelector()
        ligand_sel.set_residue_name3(ligand_resname)
        shell_sel = NeighborhoodResidueSelector(ligand_sel, shell_radius, True)

        movemap_factory = MoveMapFactory()
        movemap_factory.all_bb(False)
        movemap_factory.all_chi(False)
        movemap_factory.add_bb_action(move_map_action.mm_enable, shell_sel)
        movemap_factory.add_chi_action(move_map_action.mm_enable, shell_sel)

        coord_gen = CoordinateConstraintGenerator()
        coord_gen.set_residue_selector(shell_sel)
        coord_gen.set_sd(1.0 / max(constraint_weight, 1e-6))
        add_csts = AddConstraints()
        add_csts.add_generator(coord_gen)
        add_csts.apply(pose)
    elif mode == "full":
        movemap = pyrosetta.MoveMap()
        movemap.set_bb(True)
        movemap.set_chi(True)
    else:
        raise ValueError(f"Unknown FastRelax mode: {mode!r}")

    scorefxn = pyrosetta.create_score_function(scorefunction)
    if mode == "ligand_focused":
        from pyrosetta.rosetta.core.scoring import ScoreType
        scorefxn.set_weight(ScoreType.coordinate_constraint, constraint_weight)

    relax = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn)
    if movemap_factory is not None:
        relax.set_movemap_factory(movemap_factory)
    elif movemap is not None:
        relax.set_movemap(movemap)
    relax.apply(pose)

    pose.dump_pdb(spec["out_pdb"])
    return {"relaxed_path": spec["out_pdb"], "score": float(scorefxn(pose))}


def main() -> int:
    try:
        with open(sys.argv[1]) as fh:
            spec = json.load(fh)
        result = _run(spec)
    except Exception as e:  # noqa: BLE001 - surface any failure as JSON
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
