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
