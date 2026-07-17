from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Literal, Union

import pandas as pd

from .step import Step

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Map the user-facing model size to the upstream ESM2 model identifier that
# the `squidly` CLI expects as its second positional argument
# (validated at squidly/__main__.py:206).
_ESM2_MODELS = {
    "3B": "esm2_t36_3B_UR50D",
    "15B": "esm2_t48_15B_UR50D",
}

# Minimal sanity check for sequence content. The squidly CLI + ESM2 will fail
# on anything that isn't a bare upper-case amino-acid string, so we reject
# early with a clear error instead of letting it surface as a deep subprocess
# traceback.
_AA_PATTERN = re.compile(r"^[A-Z*\-]+$")


def _normalize_residues(value) -> str:
    """Coerce a Squidly residue prediction into a clean pipe-delimited string.

    Accepts None, NaN, list/tuple, or string. Returns '' for empty/missing.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple)):
        parts = []
        for x in value:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                continue
            s = str(x).strip()
            if s == "" or s.lower() in ("nan", "none"):
                continue
            parts.append(s)
        return "|".join(parts)
    s = str(value).strip()
    if s.lower() in ("nan", "none", "[]", ""):
        return ""
    return s


class Squidly(Step):
    """Thin Filterzyme wrapper around `enzymetk.ActiveSitePred`.

    Single responsibility: invoke the upstream `squidly` CLI via enzymetk and
    return its predictions on the full input DataFrame, with the upstream
    `Squidly_Ensemble_Residues` column renamed to the canonical
    `Squidly_CR_Position` column that `pipeline_v2` consumes.

    This step does NOT merge user-supplied `vina_residues` or drop entries
    with no residues -- that policy lives in
    `pipeline_v2._catalytic_residue_prediction`.

    The `squidly` CLI must be on `$PATH` in the current environment. The
    upstream `enzymetk.ActiveSitePred` defaults to wrapping the invocation in
    `conda run -n enzymetk`, which would fail because `squidly` is installed
    in the `structurezyme` env, not `enzymetk`. We therefore pass `env_name=None`
    so the CLI runs directly in the current shell.
    """

    def __init__(
        self,
        sequence_col: str = "Sequence",
        id_col: str = "Entry",
        model_size: Literal["3B", "15B"] = "3B",
        as_threshold: float | None = None,
        mean_prob: float | None = None,
        mean_var: float | None = None,
        single_model: bool = False,
        cpu: bool = False,
        iterative: bool = False,
        num_threads: int = 1,
        dedup_by_sequence: bool = True,
        tmp_dir: Union[str, Path, None] = None,
    ):
        self.sequence_col = sequence_col
        self.id_col = id_col
        if model_size not in _ESM2_MODELS:
            raise ValueError(
                f"model_size must be '3B' or '15B', got {model_size!r}"
            )
        self.model_size = model_size
        self.esm2_model = _ESM2_MODELS[model_size]
        self.as_threshold = as_threshold
        self.mean_prob = mean_prob
        self.mean_var = mean_var
        self.single_model = single_model
        self.cpu = cpu
        self.iterative = iterative
        self.num_threads = num_threads or 1
        self.dedup_by_sequence = dedup_by_sequence
        self.tmp_dir = Path(tmp_dir) if tmp_dir is not None else None

    def _build_cli_args(self) -> list[str]:
        """Build the `args` list forwarded to `enzymetk.ActiveSitePred`."""
        args: list[str] = []
        if self.single_model:
            args.append("--single-model")
        if self.cpu:
            args.append("--cpu")
        if self.iterative:
            args.append("--iterative")
        if self.as_threshold is not None:
            args.extend(["--as-threshold", str(self.as_threshold)])
        if self.mean_prob is not None:
            args.extend(["--mean-prob", str(self.mean_prob)])
        if self.mean_var is not None:
            args.extend(["--mean-var", str(self.mean_var)])
        return args

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = [
            c for c in (self.id_col, self.sequence_col) if c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Squidly input DataFrame is missing required columns: {missing}"
            )
        for entry, seq in df[[self.id_col, self.sequence_col]].values:
            if (
                seq is None
                or (isinstance(seq, float) and pd.isna(seq))
                or str(seq).strip() == ""
            ):
                raise ValueError(
                    f"Empty sequence for entry {entry!r}; cannot run Squidly."
                )
            s = str(seq).strip()
            if not _AA_PATTERN.match(s):
                raise ValueError(
                    f"Sequence for entry {entry!r} contains non-amino-acid "
                    f"characters; cannot run Squidly. First 20 chars: {s[:20]!r}"
                )

    def _build_representatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return one row per unique sequence so ESM2 embeds each sequence once."""
        if self.dedup_by_sequence:
            reps = (
                df[[self.id_col, self.sequence_col]]
                .drop_duplicates(subset=self.sequence_col, keep="first")
                .reset_index(drop=True)
            )
        else:
            reps = df[[self.id_col, self.sequence_col]].reset_index(drop=True)
        return reps

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not shutil.which("squidly"):
            raise RuntimeError(
                "The `squidly` CLI was not found on $PATH. Catalytic-residue "
                "prediction requires it. See "
                "`1_analysis-planning/3_setup-sanity-check/10_squidly_install.md` "
                "for install instructions."
            )
        self._validate_input(df)

        reps = self._build_representatives(df)

        # Lazy import so `import structurezyme` doesn't drag in enzymetk/torch.
        from enzymetk.predict_catalyticsite_step import ActiveSitePred

        pred = ActiveSitePred(
            id_col=self.id_col,
            seq_col=self.sequence_col,
            num_threads=self.num_threads,
            esm2_model=self.esm2_model,
            tmp_dir=str(self.tmp_dir) if self.tmp_dir is not None else None,
            args=self._build_cli_args() or None,
            env_name=None,
        )
        df_pred = pred.execute(reps)

        # Upstream pkl columns (verified in squidly/__main__.py:327-406):
        #   label, Squidly_Ensemble_Residues, Squidly_CR_Position (ensemble
        #   path only), Sequence, mean, entropy, variance, all_AS_probs_*.
        # We rename label -> the user's id_col so we can join back to reps,
        # and Squidly_Ensemble_Residues -> Squidly_CR_Position (the canonical
        # name pipeline_v2 reads). If the ensemble path already added
        # Squidly_CR_Position, we overwrite it from Squidly_Ensemble_Residues
        # so the two never disagree.
        if "label" in df_pred.columns:
            df_pred = df_pred.rename(columns={"label": self.id_col})

        residues_col = "Squidly_Ensemble_Residues"
        if residues_col in df_pred.columns:
            df_pred["Squidly_CR_Position"] = df_pred[residues_col].apply(_normalize_residues)
        elif "Squidly_CR_Position" in df_pred.columns:
            df_pred["Squidly_CR_Position"] = df_pred["Squidly_CR_Position"].apply(_normalize_residues)
        else:
            df_pred["Squidly_CR_Position"] = ""

        # Surface ensemble uncertainty columns for traceability if present.
        for src, dst in (
            ("mean", "squidly_mean"),
            ("variance", "squidly_variance"),
            ("entropy", "squidly_entropy"),
        ):
            if src in df_pred.columns:
                df_pred[dst] = df_pred[src]

        # Broadcast predictions from the deduped representatives back to the
        # full input DataFrame, keyed by Sequence. This correctly handles
        # duplicate sequences that share an Entry or have different Entries,
        # which the previous Entry-keyed lookup in pipeline_v2 did not.
        seq_to_residues = dict(
            zip(
                reps[self.sequence_col],
                reps[self.id_col].map(
                    dict(zip(df_pred[self.id_col], df_pred["Squidly_CR_Position"]))
                ),
            )
        )

        out = df.copy()
        out["Squidly_CR_Position"] = out[self.sequence_col].map(seq_to_residues).fillna("")
        out["Squidly_CR_Position"] = out["Squidly_CR_Position"].apply(_normalize_residues)

        # Attach the optional traceability columns too, keyed by Sequence.
        for dst in ("squidly_mean", "squidly_variance", "squidly_entropy"):
            if dst in df_pred.columns:
                seq_to_val = dict(
                    zip(
                        reps[self.sequence_col],
                        reps[self.id_col].map(
                            dict(zip(df_pred[self.id_col], df_pred[dst]))
                        ),
                    )
                )
                out[dst] = out[self.sequence_col].map(seq_to_val)

        return out
