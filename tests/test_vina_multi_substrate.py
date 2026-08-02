# tests/test_vina_multi_substrate.py
import pandas as pd
import structurezyme.step_runners as sr


def _ctx(tmp_path, mode):
    class _Ctx:
        class config:
            class runtime:
                num_threads = 1
            multi_substrate_mode = mode
        def step_dir(self, name):
            d = tmp_path / name
            d.mkdir(parents=True, exist_ok=True)
            return d
    return _Ctx()


class _Spec:
    name = "vina"


def test_together_multi_substrate_skips_vina(tmp_path, monkeypatch):
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.c1ccncc1"],
                       "catalytic_residues": ["1|2"],
                       "chai_dir": ["/x"], "boltz_dir": ["/y"]})
    monkeypatch.setattr(sr, "_first_input", lambda ctx, spec: df, raising=True)
    monkeypatch.setattr(sr, "_docking_dir", lambda ctx: tmp_path, raising=True)
    out = sr.run_vina(_ctx(tmp_path, "together"), _Spec())
    assert list(out["Entry"]) == ["P1"]
    assert out["vina_dir"].isna().all()
    assert (tmp_path / "vina.pkl").is_file()


def test_together_single_substrate_does_not_skip(tmp_path, monkeypatch):
    # single substrate in together mode: guard must NOT short-circuit.
    # We assert the skip branch is not taken by checking the function proceeds
    # past the guard (it will raise later trying to import Vina/real docking),
    # so we only verify the guard predicate here via the helper.
    from structurezyme.step_runners import _together_skips_vina
    df1 = pd.DataFrame({"Entry": ["P1"], "substrate_smiles": ["CCO"]})
    df2 = pd.DataFrame({"Entry": ["P1"], "substrate_smiles": ["CCO.O"]})
    assert _together_skips_vina("together", df1) is False
    assert _together_skips_vina("together", df2) is True
    assert _together_skips_vina("off", df2) is False
