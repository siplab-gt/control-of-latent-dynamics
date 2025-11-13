"""Analyze control experiment data and include fSNN and model fit analyses."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from snakemake.script import snakemake

from src.helpers import setup_debugger, setup_logging
from src.scripts.ctrl_expts import CtrlData
from src.scripts.model_fit import ModelFitPredData
from src.scripts.sys_id_expts import ExptData


def nrmse(R, X, Xbaseline):
    """normalize by baseline errors (null control)"""
    assert R.shape == X.shape == Xbaseline.shape, (
        f"Shape mismatch: {R.shape}, {X.shape}, {Xbaseline.shape}"
    )
    errors = X - R
    baseline_errors = Xbaseline - R
    return np.linalg.norm(errors) / np.linalg.norm(baseline_errors)


def is_empty(arr):
    return arr is None or arr.size == 0


def analyze_ctrl_data(
    cond_params,
    ctrl_data: CtrlData,
    null_ctrl_data: CtrlData,
    fsnn_data_analysis_file,
    model_fit_analysis_file,
    outfile,
):
    results = {**cond_params}
    eval_mask = np.concat(
        [np.arange(start, end) for start, end in ctrl_data.i_trial_boundaries]
    )
    assert np.all(ctrl_data.i_trial_boundaries == null_ctrl_data.i_trial_boundaries)

    def r2_score_masked(y_true, y_pred):
        if is_empty(y_true) or is_empty(y_pred):
            return np.nan
        return r2_score(y_true[eval_mask], y_pred[eval_mask])

    def q_track_masked(ref, achieved, baseline):
        if is_empty(ref) or is_empty(achieved) or is_empty(baseline):
            return np.nan
        return 1 - nrmse(ref[eval_mask], achieved[eval_mask], baseline[eval_mask])

    # R2
    results["R2_track_Zref_Z"] = r2_score_masked(ctrl_data.Zref, ctrl_data.Z)
    results["R2_track_Xref_X_fsnn"] = r2_score_masked(ctrl_data.Xref_fsnn, ctrl_data.X)
    results["R2_track_Xref_Xhat_fsnn"] = r2_score_masked(
        ctrl_data.Xref_fsnn, ctrl_data.Xhat_fsnn
    )
    results["R2_track_Xref_Xhat_lds"] = r2_score_masked(
        ctrl_data.Xref_lds, ctrl_data.Xhat_lds
    )

    # Q
    results["Q_track_Zref_Z"] = q_track_masked(
        ctrl_data.Zref, ctrl_data.Z, null_ctrl_data.Z
    )
    results["Q_track_Xref_X_fsnn"] = q_track_masked(
        ctrl_data.Xref_fsnn, ctrl_data.X, null_ctrl_data.X
    )
    results["Q_track_Xref_Xhat_fsnn"] = q_track_masked(
        ctrl_data.Xref_fsnn, ctrl_data.Xhat_fsnn, null_ctrl_data.Xhat_fsnn
    )
    results["Q_track_Xref_Xhat_lds"] = q_track_masked(
        ctrl_data.Xref_lds, ctrl_data.Xhat_lds, null_ctrl_data.Xhat_lds
    )

    results["ctrl_compute_latency_mean_ms"] = ctrl_data.compute_latency_mean_ms
    results["ctrl_compute_latency_std_ms"] = ctrl_data.compute_latency_std_ms
    results["ctrl_total_latency_ms"] = ctrl_data.total_latency_ms

    results |= pd.read_csv(fsnn_data_analysis_file).iloc[0].to_dict()
    results |= pd.read_csv(model_fit_analysis_file).iloc[0].to_dict()
    print(results)

    df = pd.DataFrame([results])
    print(f"Saving analysis results to {outfile}")
    df.to_csv(outfile, index=False)


if __name__ == "__main__" and "snakemake" in locals():
    fh = setup_logging(snakemake.config, snakemake.log[0])
    # setup_debugger()

    ctrl_data = CtrlData.load(snakemake.input.ctrl_data)
    null_ctrl_data = CtrlData.load(snakemake.input.null_ctrl_data)
    analyze_ctrl_data(
        snakemake.wildcards,
        ctrl_data,
        null_ctrl_data,
        snakemake.input.fsnn_data_analysis,
        snakemake.input.model_fit_analysis,
        snakemake.output[0],
    )
