# tests/test_deprecation_shim.py
import warnings


def test_filterzyme_shim_warns_and_reexports():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import filterzyme
        from filterzyme.pipeline import Pipeline  # noqa: F401
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    import structurezyme
    assert filterzyme.__version__ == structurezyme.__version__
