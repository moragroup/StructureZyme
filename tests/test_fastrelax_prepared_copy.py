"""Relaxed poses must land in preparedfiles_for_superimposition/.

When fastrelax is enabled it renames poses to `<stem>_relaxed.pdb` and rewrites
`docked_structure` to those names. The downstream analysis steps
(geometric_filter, fpocket, ligand_sasa, plip, placer) look up
`preparedfiles_for_superimposition/<docked_structure>.pdb`. That prepared dir is
populated by prepare_files with the UN-relaxed names only, so without copying
the relaxed pdbs across, every downstream lookup 404s and the step silently
skips all structures. `_publish_relaxed_to_prepared` bridges that gap.
"""
from pathlib import Path

from structurezyme.step_runners import _publish_relaxed_to_prepared


def test_copies_relaxed_pdbs_into_prepared(tmp_path):
    fastrelax_dir = tmp_path / "fastrelax"
    prepared_dir = tmp_path / "preparedfiles_for_superimposition"
    fastrelax_dir.mkdir()
    prepared_dir.mkdir()
    # prepared dir starts with the un-relaxed poses (as prepare_files writes)
    (prepared_dir / "P41365_0_chai.pdb").write_text("ORIG")
    # fastrelax writes relaxed variants in its own dir
    (fastrelax_dir / "P41365_0_chai_relaxed.pdb").write_text("RELAXED0")
    (fastrelax_dir / "P41365_model_0_boltz_relaxed.pdb").write_text("RELAXEDB")

    n = _publish_relaxed_to_prepared(fastrelax_dir, prepared_dir)

    assert n == 2
    # relaxed poses now resolvable under their _relaxed names in prepared dir
    assert (prepared_dir / "P41365_0_chai_relaxed.pdb").read_text() == "RELAXED0"
    assert (prepared_dir / "P41365_model_0_boltz_relaxed.pdb").read_text() == "RELAXEDB"
    # original un-relaxed file left intact
    assert (prepared_dir / "P41365_0_chai.pdb").read_text() == "ORIG"


def test_no_relaxed_files_is_noop(tmp_path):
    fastrelax_dir = tmp_path / "fastrelax"
    prepared_dir = tmp_path / "prepared"
    fastrelax_dir.mkdir()
    prepared_dir.mkdir()
    assert _publish_relaxed_to_prepared(fastrelax_dir, prepared_dir) == 0


def test_missing_fastrelax_dir_is_noop(tmp_path):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    assert _publish_relaxed_to_prepared(tmp_path / "nope", prepared_dir) == 0
