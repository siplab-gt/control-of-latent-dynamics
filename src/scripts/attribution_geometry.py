"""NULL-trial geometry the attribution controller arms are defined against (EVAL_PLAN P1).

Two of P1's cost terms are stated in units the plant supplies rather than the model:

- `Ω` weights readout directions by how little the plant visits them without input, so the
  unified-LDS arm can be given a two-sided setpoint on its atypicality share.
- `P₁` is the natural block's own NULL-trial spread, so the (n′) term measures `Δx1` in units of
  intrinsic variability rather than raw state norm.

Both come from a NULL cell — a null *controller*, not a null task, so the statistics are of the
network performing the task under endogenous drive with no opto. Computing them in their own rule
keeps one geometry shared across every α cell of a sweep, instead of re-deriving it per cell from
a NULL twin that would then have to exist at every α.
"""
import numpy as np
from snakemake.script import snakemake

from src.helpers import load_pkl_gz, preprocess_params, setup_debugger, setup_logging
from src.scripts.ctrl_expts import CtrlData

LAMBDA_REF_FRACTION = 0.1
"""`lambda_ref` as a fraction of the median NULL eigenvalue. The controller needs one committed
value. The sweep over a decade lives in the offline metrics, where re-deriving it is free."""


def build_geometry(null_ctrl_data, n1):
    """Covariances and the graded-weight scale, from a NULL control cell.

    Args:
        null_ctrl_data: `CtrlData` from a `ctrl_type=NULL` run.
        n1: natural-block dimension, or 0 for a unified LDS (no `P₁`).

    Returns:
        dict of arrays to save: `sigma_null`, `lambda_ref`, and `P1` when `n1 > 0`.
    """
    eval_mask = np.concat(
        [np.arange(start, end) for start, end in null_ctrl_data.i_trial_boundaries]
    )
    X_null = np.asarray(null_ctrl_data.X, dtype=float)[eval_mask]
    sigma_null = np.atleast_2d(np.cov(X_null, rowvar=False))

    eigenvalues = np.clip(np.linalg.eigvalsh(sigma_null), 0.0, None)
    out = {
        "sigma_null": sigma_null,
        "lambda_ref": np.asarray(LAMBDA_REF_FRACTION * float(np.median(eigenvalues))),
    }
    if n1 > 0:
        x1_null = np.asarray(null_ctrl_data.Xhat_lds, dtype=float)[eval_mask, :n1]
        out["P1"] = np.atleast_2d(np.cov(x1_null, rowvar=False))
    return out


if __name__ == "__main__" and "snakemake" in locals():
    setup_logging(snakemake.config, snakemake.log[0])
    setup_debugger(snakemake.config["debug"])
    preprocess_params(snakemake.config)

    lds = load_pkl_gz(snakemake.input.lds_model)
    np.savez(
        snakemake.output[0],
        **build_geometry(
            CtrlData.load(snakemake.input.null_ctrl_data),
            lds.A11.shape[0] if hasattr(lds, "A11") else 0,
        ),
    )
