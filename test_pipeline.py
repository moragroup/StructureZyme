"""Minimal smoke test for filterzyme.pipeline_v2.Pipeline.

This replaces the old test_pipeline.py which imported the long-gone
`filtering_pipeline` package. It verifies that the v2 pipeline can be
constructed and that the Squidly step is reachable, without requiring a
GPU or Boltz cache (the actual `.run()` is not executed).

To run the full pipeline end-to-end, use the README quick-start on a GPU
node with a Boltz cache directory.
"""
import pandas as pd
import pytest


def test_pipeline_v2_import():
    """Pipeline and Docking can be imported from pipeline_v2."""
    from filterzyme.pipeline_v2 import Pipeline, Docking
    assert Pipeline is not None
    assert Docking is not None


def test_docking_accepts_squidly_kwargs():
    """Docking.__init__ accepts the squidly_* kwargs added in Phase B."""
    from filterzyme.pipeline_v2 import Docking
    import inspect
    sig = inspect.signature(Docking.__init__)
    params = sig.parameters
    assert "squidly_model_size" in params
    assert "squidly_as_threshold" in params
    assert "squidly_num_threads" in params
    assert params["squidly_model_size"].default == "3B"
    assert params["squidly_as_threshold"].default is None


def test_pipeline_accepts_squidly_kwargs():
    """Pipeline.__init__ forwards the squidly_* kwargs to Docking."""
    from filterzyme.pipeline_v2 import Pipeline
    import inspect
    sig = inspect.signature(Pipeline.__init__)
    params = sig.parameters
    assert "squidly_model_size" in params
    assert "squidly_as_threshold" in params
    assert "squidly_num_threads" in params


def test_pipeline_construction():
    """Pipeline can be constructed with a minimal DataFrame."""
    from filterzyme.pipeline_v2 import Pipeline
    df = pd.DataFrame({
        "Entry": ["enzyme_1"],
        "Sequence": ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"],
        "substrate_smiles": ["CCOC(=O)C"],
        "substrate_name": ["ethyl_acetate"],
        "substrate_moiety": ["[C](=O)([O])([O])"],
    })
    pipeline = Pipeline(
        df=df,
        boltz_cache_dir="/tmp/fake_boltz_cache",
        base_output_dir="/tmp/filterzyme_test_output",
    )
    assert pipeline is not None
    assert pipeline.squidly_model_size == "3B"


def test_pipeline_accepts_placer_kwargs():
    """Pipeline.__init__ accepts run_placer + 4 placer_* kwargs with correct defaults."""
    from filterzyme.pipeline_v2 import Pipeline
    import inspect
    sig = inspect.signature(Pipeline.__init__)
    params = sig.parameters
    assert "run_placer" in params
    assert params["run_placer"].default is False
    assert "placer_predict_ligand" in params
    assert params["placer_predict_ligand"].default is None
    assert "placer_nsamples" in params
    assert params["placer_nsamples"].default == 50
    assert "placer_rerank" in params
    assert params["placer_rerank"].default == "prmsd"
    assert "placer_conda_env" in params
    assert params["placer_conda_env"].default == "placer_env"


def test_pipeline_run_placer_without_ligand_raises():
    """Pipeline(run_placer=True) without placer_predict_ligand raises ValueError."""
    from filterzyme.pipeline_v2 import Pipeline
    import pandas as pd
    df = pd.DataFrame({
        "Entry": ["e1"],
        "Sequence": ["MK"],
        "substrate_smiles": ["CC"],
        "substrate_name": ["x"],
        "substrate_moiety": ["[C]"],
    })
    with pytest.raises(ValueError, match="placer_predict_ligand"):
        Pipeline(
            df=df,
            boltz_cache_dir="/tmp/fake",
            base_output_dir="/tmp/filterzyme_test_placer_missing_ligand",
            run_placer=True,
        )


def test_pipeline_run_placer_false_skips_placer(tmp_path, monkeypatch):
    """When run_placer=False (default), PLACER.execute is never invoked."""
    from filterzyme.pipeline_v2 import Pipeline
    from filterzyme.steps import PLACER_step
    import filterzyme.pipeline_v2 as pv2
    import pandas as pd

    # Stub upstream steps so run() short-circuits harmlessly.
    monkeypatch.setattr(pv2.Docking, "run", lambda self: None)
    monkeypatch.setattr(pv2.Superimposition, "run", lambda self: None)
    monkeypatch.setattr(pv2.GeometricFilters, "run", lambda self: None)

    # Create the pkl file that Superimposition would have written (read by
    # Pipeline.run before GeometricFilters is instantiated).
    superimp_dir = tmp_path / "superimposition"
    superimp_dir.mkdir(parents=True)
    pd.DataFrame({"Entry": ["e1"]}).to_pickle(superimp_dir / "ligandRMSD.pkl")

    # Create the pkl file that GeometricFilters would have written.
    geo_dir = tmp_path / "geometricfiltering"
    geo_dir.mkdir(parents=True)
    stub_df = pd.DataFrame({
        "Entry": ["e1"], "docked_structure": ["e1"],
        "is_best": [True], "best_method": ["chai"],
    })
    stub_df.to_pickle(geo_dir / "structural_features_final.pkl")

    # Spy on PLACER.execute
    called = {"n": 0}
    def spy(self, df):
        called["n"] += 1
        return df
    monkeypatch.setattr(PLACER_step.PLACER, "execute", spy)

    df = pd.DataFrame({
        "Entry": ["e1"], "Sequence": ["MK"],
        "substrate_smiles": ["CC"], "substrate_name": ["x"],
        "substrate_moiety": ["[C]"],
    })
    Pipeline(
        df=df,
        boltz_cache_dir="/tmp/fake",
        base_output_dir=str(tmp_path),
    ).run()

    assert called["n"] == 0, "PLACER.execute must not run when run_placer=False"


def test_pipeline_run_placer_true_invokes_execute(tmp_path, monkeypatch):
    """When run_placer=True, PLACER.execute is called once with the geo pkl DataFrame."""
    from filterzyme.pipeline_v2 import Pipeline
    from filterzyme.steps import PLACER_step
    import filterzyme.pipeline_v2 as pv2
    import pandas as pd

    monkeypatch.setattr(pv2.Docking, "run", lambda self: None)
    monkeypatch.setattr(pv2.Superimposition, "run", lambda self: None)
    monkeypatch.setattr(pv2.GeometricFilters, "run", lambda self: None)

    # Stub PLACER __init__ so it doesn't check for the real script.
    monkeypatch.setattr(
        PLACER_step.PLACER, "__init__",
        lambda self, **kwargs: setattr(self, "_kwargs", kwargs) or None,
    )

    # Create the pkl file that Superimposition would have written (read by
    # Pipeline.run before GeometricFilters is instantiated).
    superimp_dir = tmp_path / "superimposition"
    superimp_dir.mkdir(parents=True)
    pd.DataFrame({"Entry": ["e1"]}).to_pickle(superimp_dir / "ligandRMSD.pkl")

    geo_dir = tmp_path / "geometricfiltering"
    geo_dir.mkdir(parents=True)
    stub_df = pd.DataFrame({
        "Entry": ["e1", "e2"], "docked_structure": ["e1", "e2"],
        "is_best": [True, False], "best_method": ["chai", "boltz"],
    })
    stub_df.to_pickle(geo_dir / "structural_features_final.pkl")

    calls = []
    def spy(self, df):
        calls.append(df.copy())
        df["placer_prmsd"] = [0.5, None]
        return df
    monkeypatch.setattr(PLACER_step.PLACER, "execute", spy)

    df = pd.DataFrame({
        "Entry": ["e1"], "Sequence": ["MK"],
        "substrate_smiles": ["CC"], "substrate_name": ["x"],
        "substrate_moiety": ["[C]"],
    })
    Pipeline(
        df=df,
        boltz_cache_dir="/tmp/fake",
        base_output_dir=str(tmp_path),
        run_placer=True,
        placer_predict_ligand="A-HEM-154",
    ).run()

    assert len(calls) == 1
    assert list(calls[0]["Entry"]) == ["e1", "e2"]

    # Merged output must be persisted back to structural_features_final.pkl
    merged = pd.read_pickle(geo_dir / "structural_features_final.pkl")
    assert "placer_prmsd" in merged.columns
    assert merged.loc[merged["Entry"] == "e1", "placer_prmsd"].iloc[0] == 0.5
