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
    assert "placer_env_path" in params
    assert params["placer_env_path"].default == "/mnt/labs/data/mora/software/PLACER/env"


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


def test_superimposition_accepts_fastrelax_kwargs():
    """Superimposition.__init__ accepts the fastrelax_* kwargs + ligand_resname."""
    from filterzyme.pipeline_v2 import Superimposition
    import inspect
    sig = inspect.signature(Superimposition.__init__)
    params = sig.parameters
    assert "run_fastrelax" in params
    assert params["run_fastrelax"].default is False
    assert "fastrelax_mode" in params
    assert params["fastrelax_mode"].default == "ligand_focused"
    assert "fastrelax_top_k" in params
    assert params["fastrelax_top_k"].default == 2
    assert "fastrelax_drop_unrelaxed" in params
    assert params["fastrelax_drop_unrelaxed"].default is True
    assert "fastrelax_shell_radius" in params
    assert params["fastrelax_shell_radius"].default == 8.0
    assert "fastrelax_constraint_weight" in params
    assert params["fastrelax_constraint_weight"].default == 1.0
    assert "fastrelax_scorefunction" in params
    assert params["fastrelax_scorefunction"].default == "ref2015"
    assert "ligand_resname" in params
    assert params["ligand_resname"].default == "LIG"


def _superimposition_run_stubs(monkeypatch):
    """Monkeypatch Superimposition's heavy per-step helpers to lightweight
    stubs so `.run()` can be exercised in-process without file I/O."""
    from filterzyme.pipeline_v2 import Superimposition
    monkeypatch.setattr(
        Superimposition, "_prepare_files_for_superimposition",
        lambda self: pd.DataFrame({"Entry": ["e1"]}),
    )
    monkeypatch.setattr(
        Superimposition, "_superimposition",
        lambda self, df: df,
    )
    monkeypatch.setattr(
        Superimposition, "_proteinRMSD",
        lambda self, df: (df, df),
    )
    monkeypatch.setattr(
        Superimposition, "_ligandRMSD",
        lambda self, df: (df, df),
    )


def test_superimposition_run_fastrelax_skips_when_disabled(tmp_path, monkeypatch):
    """When run_fastrelax=False, `_run_fastrelax` must NOT be called."""
    from unittest.mock import MagicMock
    from filterzyme.pipeline_v2 import Superimposition
    _superimposition_run_stubs(monkeypatch)
    mock_fr = MagicMock()
    monkeypatch.setattr(Superimposition, "_run_fastrelax", mock_fr)
    sup = Superimposition(
        maxMatches=1000,
        input_dir=str(tmp_path),
        output_dir=str(tmp_path),
        run_fastrelax=False,
    )
    sup.run()
    assert mock_fr.call_count == 0


def test_superimposition_run_fastrelax_called_when_enabled(tmp_path, monkeypatch):
    """When run_fastrelax=True, `_run_fastrelax` must be called with df_prep,
    and its return value must flow into `_superimposition`."""
    from unittest.mock import MagicMock
    from filterzyme.pipeline_v2 import Superimposition
    _superimposition_run_stubs(monkeypatch)

    df_prep = pd.DataFrame({"Entry": ["e1"]})
    df_after_relax = pd.DataFrame({"Entry": ["e1"], "relaxed": [True]})

    monkeypatch.setattr(
        Superimposition, "_prepare_files_for_superimposition",
        lambda self: df_prep,
    )
    mock_fr = MagicMock(return_value=df_after_relax)
    monkeypatch.setattr(Superimposition, "_run_fastrelax", mock_fr)

    seen = {}
    def stub_superimposition(self, df):
        seen["df"] = df
        return df
    monkeypatch.setattr(Superimposition, "_superimposition", stub_superimposition)

    sup = Superimposition(
        maxMatches=1000,
        input_dir=str(tmp_path),
        output_dir=str(tmp_path),
        run_fastrelax=True,
    )
    sup.run()
    assert mock_fr.call_count == 1
    # MagicMock, when installed on the class, is a descriptor and behaves
    # like a bound method: `self` is swallowed and does not appear in args.
    assert mock_fr.call_args.args[0] is df_prep
    assert seen["df"] is df_after_relax


def test_run_fastrelax_constructs_fastrelax_step_correctly(tmp_path, monkeypatch):
    """`_run_fastrelax(df_prep)` must construct a FastRelax step with the
    Superimposition instance's fastrelax_* kwargs threaded through, call
    `.execute(df_prep)`, and return execute()'s return value."""
    from unittest.mock import MagicMock
    from pathlib import Path
    import filterzyme.pipeline_v2 as pv2

    df_prep = pd.DataFrame({"Entry": ["e1"]})
    df_after = pd.DataFrame({"Entry": ["e1"], "relaxed": [True]})

    mock_instance = MagicMock()
    mock_instance.execute.return_value = df_after
    MockFastRelax = MagicMock(return_value=mock_instance)
    monkeypatch.setattr(pv2, "FastRelax", MockFastRelax)

    sup = pv2.Superimposition(
        maxMatches=1000,
        input_dir=str(tmp_path),
        output_dir=str(tmp_path),
        run_fastrelax=True,
        fastrelax_top_k=3,
        fastrelax_mode="full",
        fastrelax_drop_unrelaxed=False,
        fastrelax_shell_radius=10.0,
        fastrelax_constraint_weight=0.5,
        fastrelax_scorefunction="beta_nov16",
        ligand_resname="ABC",
        num_threads=4,
    )
    result = sup._run_fastrelax(df_prep)

    MockFastRelax.assert_called_once()
    _, kwargs = MockFastRelax.call_args
    assert kwargs["top_k"] == 3
    assert kwargs["mode"] == "full"
    assert kwargs["drop_unrelaxed"] is False
    assert kwargs["shell_radius"] == 10.0
    assert kwargs["constraint_weight"] == 0.5
    assert kwargs["scorefunction"] == "beta_nov16"
    assert kwargs["ligand_resname"] == "ABC"
    assert kwargs["num_threads"] == 4
    assert Path(kwargs["output_dir"]) == Path(tmp_path) / "fastrelax"

    mock_instance.execute.assert_called_once_with(df_prep)
    assert result is df_after
