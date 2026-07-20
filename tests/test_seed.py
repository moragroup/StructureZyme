# tests/test_seed.py
import pytest
from structurezyme.config import RunConfig, PathsConfig


def _cfg(**paths):
    base = dict(output_root="/tmp/o", boltz_cache_dir="/tmp/b", input_csv="/tmp/in.csv")
    base.update(paths)
    return RunConfig(paths=PathsConfig(**base))


def test_validate_paths_requires_input_csv():
    cfg = _cfg(input_csv="")
    with pytest.raises(ValueError) as e:
        cfg.validate_paths()
    assert "input_csv" in str(e.value)


def test_validate_paths_ok_with_input_csv():
    assert _cfg().validate_paths() is not None


import pandas as pd
from structurezyme.runner import (
    _load_input_frame, missing_input_columns, REQUIRED_INPUT_COLUMNS,
)


def test_load_input_frame_csv(tmp_path):
    p = tmp_path / "in.csv"
    pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"]}).to_csv(p, index=False)
    df = _load_input_frame(str(p))
    assert list(df.columns) == ["Sequence", "substrate_smiles", "Entry"]


def test_load_input_frame_pickle(tmp_path):
    p = tmp_path / "in.pkl"
    pd.DataFrame({"Sequence": ["M"], "cofactor_moiety": [None]}).to_pickle(p)
    df = _load_input_frame(str(p))
    assert df["cofactor_moiety"].iloc[0] is None


def test_missing_columns_base_only():
    df = pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"]})
    assert missing_input_columns(df, enabled=set()) == {}


def test_missing_columns_vina():
    df = pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"]})
    miss = missing_input_columns(df, enabled={"vina"})
    assert miss.get("vina") == ["vina_residues"]


def test_missing_columns_geometric_filter():
    df = pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"]})
    miss = missing_input_columns(df, enabled={"geometric_filter"})
    assert miss.get("geometric_filter") == ["substrate_moiety"]


from structurezyme.config import RuntimeConfig, StepsConfig, StepConfig
from structurezyme.runner import Runner


def _runner_cfg(tmp_path, input_csv, **steps):
    return RunConfig(
        paths=PathsConfig(output_root=str(tmp_path / "out"),
                          boltz_cache_dir=str(tmp_path / "bcache"),
                          input_csv=str(input_csv)),
        runtime=RuntimeConfig(user="u", run_id="r1"),
        steps=StepsConfig(**steps),
    )


def _base_csv(tmp_path):
    p = tmp_path / "in.csv"
    # substrate_moiety is included because geometric_filter is enabled by
    # default in StepsConfig; without it, column validation would fail before
    # these tests can exercise the seeding behaviour they target.
    pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"],
                  "substrate_moiety": ["C"]}).to_csv(p, index=False)
    return p


def test_seed_writes_input_pkl(tmp_path):
    r = Runner(_runner_cfg(tmp_path, _base_csv(tmp_path)))
    r._seed_and_validate()
    seed = r.layout.checkpoint_path("_input")
    assert seed.is_file()
    assert list(pd.read_pickle(seed)["Entry"]) == ["P1"]


def test_seed_does_not_clobber_existing(tmp_path):
    r = Runner(_runner_cfg(tmp_path, _base_csv(tmp_path)))
    seed = r.layout.checkpoint_path("_input")
    pd.DataFrame({"Sequence": ["X"], "substrate_smiles": ["N"], "Entry": ["KEEP"],
                  "substrate_moiety": ["N"]}).to_pickle(seed)
    r._seed_and_validate()
    assert list(pd.read_pickle(seed)["Entry"]) == ["KEEP"]


def test_seed_raises_on_missing_enabled_column(tmp_path):
    cfg = _runner_cfg(tmp_path, _base_csv(tmp_path), vina=StepConfig(enabled=True))
    r = Runner(cfg)
    with pytest.raises(ValueError) as e:
        r._seed_and_validate()
    assert "vina" in str(e.value) and "vina_residues" in str(e.value)
