# tests/test_pipeline_adapter.py
import warnings

import pandas as pd

from structurezyme.pipeline import Pipeline


def _df():
    return pd.DataFrame({
        "Entry": ["e1"],
        "Sequence": ["MKT"],
        "substrate_smiles": ["CC"],
        "substrate_name": ["x"],
        "substrate_moiety": ["[C]"],
    })


def test_pipeline_kwargs_build_runconfig(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = Pipeline(df=_df(), boltz_cache_dir=str(tmp_path / "cache"),
                     base_output_dir=str(tmp_path / "out"), run_vina=False)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert p.config.is_enabled("vina") is False
    assert p.config.is_enabled("chai") is True


def test_adapter_maps_paths_and_step_toggles(tmp_path):
    p = Pipeline(df=_df(), boltz_cache_dir=str(tmp_path / "cache"),
                 base_output_dir=str(tmp_path / "out"),
                 run_vina=True, skip_catalytic_residue_prediction=True,
                 run_fastrelax=True, num_threads=4, esterase=1, max_matches=250)
    cfg = p.config
    assert cfg.paths.output_root == str(tmp_path / "out")
    assert cfg.paths.boltz_cache_dir == str(tmp_path / "cache")
    assert cfg.is_enabled("vina") is True
    # skip_catalytic_residue_prediction=True -> squidly still enabled but option set
    assert cfg.step_options("squidly")["skip_catalytic_residue_prediction"] is True
    assert cfg.is_enabled("fastrelax") is True
    assert cfg.runtime.num_threads == 4
    assert cfg.step_options("geometric_filter")["esterase"] == 1
    assert cfg.step_options("ligand_rmsd")["max_matches"] == 250


def test_adapter_seeds_input_checkpoint(tmp_path):
    p = Pipeline(df=_df(), boltz_cache_dir=str(tmp_path / "cache"),
                 base_output_dir=str(tmp_path / "out"))
    seed = p.layout.checkpoint_path("_input")
    assert seed.is_file()
    seeded = pd.read_pickle(seed)
    assert list(seeded["Entry"]) == ["e1"]


def test_adapter_fresh_run_id_per_instance(tmp_path):
    p1 = Pipeline(df=_df(), boltz_cache_dir=str(tmp_path / "c"),
                  base_output_dir=str(tmp_path / "o"))
    p2 = Pipeline(df=_df(), boltz_cache_dir=str(tmp_path / "c"),
                  base_output_dir=str(tmp_path / "o"))
    assert p1.config.runtime.run_id != p2.config.runtime.run_id
