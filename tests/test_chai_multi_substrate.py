# tests/test_chai_multi_substrate.py
import pandas as pd
import structurezyme.step_runners as sr


class _RecordingChai:
    last_args = None
    substrate_value_seen = None

    def __init__(self, id_col, seq_col, substrate_col, cofactor_col,
                 output_dir, num_threads):
        _RecordingChai.last_args = (id_col, seq_col, substrate_col,
                                    cofactor_col)
        self.substrate_col = substrate_col
        self.output_dir = output_dir

    def __rlshift__(self, df):  # df << step
        df = df.copy()
        # record the value at the very column the Chai step is told to read,
        # i.e. exactly what would be handed to docko's run_chai
        _RecordingChai.substrate_value_seen = df[self.substrate_col].iloc[0]
        df["output_dir"] = [str(self.output_dir)] * len(df)
        return df

    def __rshift__(self, other):  # step >> Save(...)
        return self


def test_chai_receives_intact_dotjoined_smiles(tmp_path, monkeypatch):
    # together-mode frame: two substrates packed with '.'
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.c1ccncc1"],
                       "enzyme_id": ["P1"]})

    # stub enzymetk Chai + Save so nothing docks
    import enzymetk.dock_chai_step as chai_mod
    monkeypatch.setattr(chai_mod, "Chai", _RecordingChai, raising=True)
    from structurezyme.steps import save_step
    monkeypatch.setattr(save_step, "Save",
                        lambda *a, **k: type("S", (), {"__rrshift__":
                            lambda self, d: d})(), raising=True)

    class _Ctx:
        class config:
            class runtime:
                num_threads = 1
        def step_dir(self, name):
            d = tmp_path / name
            d.mkdir(parents=True, exist_ok=True)
            return d
    ctx = _Ctx()
    monkeypatch.setattr(sr, "_first_input", lambda ctx, spec: df,
                        raising=True)
    monkeypatch.setattr(sr, "_docking_dir", lambda ctx: tmp_path,
                        raising=True)

    class _Spec:
        name = "chai"
    out = sr.run_chai(ctx, _Spec())

    # the substrate COLUMN name is what Chai is told to read; the frame still
    # holds the intact dot-joined value under that column
    assert _RecordingChai.last_args[2] == "substrate_smiles"
    assert out["substrate_smiles"].iloc[0] == "CCO.c1ccncc1"
    # and, decisively, the value AT the column the Chai step actually reads is
    # the intact dot-joined cell (this is exactly what reaches docko run_chai):
    # a future refactor that pre-splits/mangles the substrate value would fail here
    assert _RecordingChai.substrate_value_seen == "CCO.c1ccncc1"
