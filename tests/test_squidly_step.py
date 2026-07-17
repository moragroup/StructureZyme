"""Tests for `structurezyme.steps.squidly_step`.

The CLI-dependent test (`test_squidly_execute_smoke`) is skipped automatically
when the `squidly` binary is not on $PATH or its model weights are not
installed, so this file can run in CI without a GPU. The pure-logic tests
(`test_normalize_residues_*`, `test_build_cli_args`, `test_validate_input_*`,
`test_model_size_mapping`) always run.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from structurezyme.steps.squidly_step import Squidly, _normalize_residues


# A short, real serine-hydrolase-like sequence (~120 aa) with a known catalytic
# triad. Used for the CLI smoke test only.
_SMOKE_SEQ = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
    "AVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHI"
)


# --- pure-logic tests (always run) -----------------------------------------


def test_normalize_residues_none():
    assert _normalize_residues(None) == ""


def test_normalize_residues_nan():
    assert _normalize_residues(float("nan")) == ""


def test_normalize_residues_empty_string():
    assert _normalize_residues("") == ""
    assert _normalize_residues("   ") == ""


def test_normalize_residues_sentinel_strings():
    assert _normalize_residues("nan") == ""
    assert _normalize_residues("None") == ""
    assert _normalize_residues("[]") == ""


def test_normalize_residues_list():
    assert _normalize_residues([12, 45, 78]) == "12|45|78"


def test_normalize_residues_list_with_blanks():
    assert _normalize_residues([12, "", 78, None]) == "12|78"


def test_normalize_residues_pipe_string_passthrough():
    assert _normalize_residues("12|45|78") == "12|45|78"


def test_model_size_mapping_valid():
    s = Squidly(model_size="3B")
    assert s.esm2_model == "esm2_t36_3B_UR50D"
    s = Squidly(model_size="15B")
    assert s.esm2_model == "esm2_t48_15B_UR50D"


def test_model_size_mapping_invalid():
    with pytest.raises(ValueError, match="model_size"):
        Squidly(model_size="7B")


def test_build_cli_args_default_empty():
    s = Squidly()
    assert s._build_cli_args() == []


def test_build_cli_args_all_flags():
    s = Squidly(
        single_model=True,
        cpu=True,
        iterative=True,
        as_threshold=0.9,
        mean_prob=0.5,
        mean_var=0.2,
    )
    args = s._build_cli_args()
    assert "--single-model" in args
    assert "--cpu" in args
    assert "--iterative" in args
    i = args.index("--as-threshold")
    assert args[i + 1] == "0.9"
    i = args.index("--mean-prob")
    assert args[i + 1] == "0.5"
    i = args.index("--mean-var")
    assert args[i + 1] == "0.2"


def test_validate_input_missing_columns():
    s = Squidly()
    df = pd.DataFrame({"Entry": ["a"], "NotSequence": ["M"]})
    with pytest.raises(ValueError, match="missing required columns"):
        s._validate_input(df)


def test_validate_input_empty_sequence():
    s = Squidly()
    df = pd.DataFrame({"Entry": ["a"], "Sequence": [""]})
    with pytest.raises(ValueError, match="Empty sequence"):
        s._validate_input(df)


def test_validate_input_non_aa_characters():
    s = Squidly()
    df = pd.DataFrame({"Entry": ["a"], "Sequence": ["MKT1XYZ"]})
    with pytest.raises(ValueError, match="non-amino-acid"):
        s._validate_input(df)


def test_validate_input_lowercase_rejected():
    s = Squidly()
    df = pd.DataFrame({"Entry": ["a"], "Sequence": ["mkayi"]})
    with pytest.raises(ValueError, match="non-amino-acid"):
        s._validate_input(df)


def test_build_representatives_dedup():
    s = Squidly(dedup_by_sequence=True)
    df = pd.DataFrame(
        {
            "Entry": ["e1", "e2", "e3"],
            "Sequence": ["AAAA", "AAAA", "BBBB"],
        }
    )
    reps = s._build_representatives(df)
    assert len(reps) == 2
    assert list(reps["Entry"]) == ["e1", "e3"]


def test_build_representatives_no_dedup():
    s = Squidly(dedup_by_sequence=False)
    df = pd.DataFrame(
        {
            "Entry": ["e1", "e2", "e3"],
            "Sequence": ["AAAA", "AAAA", "BBBB"],
        }
    )
    reps = s._build_representatives(df)
    assert len(reps) == 3


def test_execute_missing_squidly_binary(monkeypatch):
    """If `squidly` is not on $PATH, execute() raises a clear RuntimeError."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    s = Squidly()
    df = pd.DataFrame({"Entry": ["a"], "Sequence": ["MKAIL"]})
    with pytest.raises(RuntimeError, match="squidly"):
        s.execute(df)


# --- CLI smoke test (skipped without the binary + models + GPU) ------------


def _squidly_models_installed() -> bool:
    """The `squidly` CLI needs its model weights downloaded via `squidly install`."""
    try:
        import squidly as _sq  # noqa: F401
        import os
        pkg_dir = os.path.dirname(_sq.__file__)
        return Path(pkg_dir, "models").exists()
    except Exception:
        return False


def _squidly_cli_available() -> bool:
    """True if the `squidly` binary is importable / on $PATH.

    `shutil.which` alone is insufficient when pytest is launched via an absolute
    python interpreter path without the conda env activated (the env's `bin/`
    is not on $PATH). We fall back to checking that the `squidly` package is
    importable, which implies its CLI entry point lives in the same env's
    `bin/`.
    """
    if shutil.which("squidly"):
        return True
    try:
        import squidly  # noqa: F401
        return True
    except Exception:
        return False


def _gpu_available() -> bool:
    """True if a CUDA GPU is visible to torch. The ensemble ESM2 3B model is
    impractically slow on CPU (tens of minutes per sequence), so the smoke
    tests require a GPU. Use `--cpu` + a very long timeout only if you
    specifically want to exercise the CPU path."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


_SQUIDLY_SKIP_REASON = (
    "squidly CLI, its model weights, and a CUDA GPU are all required; "
    "got binary={}, models={}, gpu={}".format(
        _squidly_cli_available(),
        _squidly_models_installed(),
        _gpu_available(),
    )
)


_SMOKE_SKIP = not (
    _squidly_cli_available() and _squidly_models_installed() and _gpu_available()
)


@pytest.mark.skipif(
    _SMOKE_SKIP,
    reason=_SQUIDLY_SKIP_REASON,
)
def test_squidly_execute_smoke():
    s = Squidly(model_size="3B", num_threads=1)
    df = pd.DataFrame({"Entry": ["test_hydrolase"], "Sequence": [_SMOKE_SEQ]})
    out = s.execute(df)

    # Entry preserved
    assert list(out["Entry"]) == ["test_hydrolase"]
    # Canonical column exists
    assert "Squidly_CR_Position" in out.columns
    # Output is a clean string (may be empty for a non-catalytic sequence, but
    # the type must be str, not NaN/None)
    val = out.loc[0, "Squidly_CR_Position"]
    assert isinstance(val, str)
    # Input sequence unchanged
    assert out.loc[0, "Sequence"] == _SMOKE_SEQ


@pytest.mark.skipif(
    _SMOKE_SKIP,
    reason=_SQUIDLY_SKIP_REASON,
)
def test_squidly_execute_dedup_broadcast():
    """Two rows with the same Sequence must both receive the same prediction."""
    s = Squidly(model_size="3B", num_threads=1)
    df = pd.DataFrame(
        {
            "Entry": ["dupe_a", "dupe_b"],
            "Sequence": [_SMOKE_SEQ, _SMOKE_SEQ],
        }
    )
    out = s.execute(df)
    assert len(out) == 2
    assert out.loc[0, "Squidly_CR_Position"] == out.loc[1, "Squidly_CR_Position"]
