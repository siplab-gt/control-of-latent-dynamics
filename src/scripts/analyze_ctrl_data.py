"""Analyze control experiment data and include fSNN and model fit analyses."""

import brian2.only as b2
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from snakemake.script import snakemake

from src.helpers import (
    load_factors_emg,
    preprocess_params,
    setup_debugger,
    setup_logging,
)
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
    cfg,
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

    # TODO: should just save conditions from fSNN in ExptData, rather than reverse-engineering them here
    if cond_params["task"] != "reaching":
        raise NotImplementedError
    i_starts = ctrl_data.i_trial_boundaries[:, 0]
    # i_ends = ctrl_data.i_trial_boundaries[:, 1]
    # u_task = ctrl_data.Utask[i_starts]
    # angle = np.arctan2(u_task[:, 1], u_task[:, 0])
    # ref_conds = angle / (2 * np.pi)

    _, emg_templates = load_factors_emg("src/fsnn/reaching_data.npz")
    n_conds = 8
    assert emg_templates.shape[0] == n_conds
    n_muscles = 2
    assert emg_templates.shape[2] == n_muscles
    # invert the first signal, as fSNN does
    emg_templates[..., 0] *= -1
    # EMG data is spaced by 10 ms
    assert 10 * b2.ms == cfg.prep_fsnn.dt * cfg.cleo_prep.subsample_factor
    # if it weren't we'd need to match timescales here
    nt = emg_templates.shape[1]

    from sklearn.neighbors import KNeighborsClassifier

    # predict reaches instead from Zref since Utask has direction input removed
    Zref = np.array(
        [ctrl_data.Zref[i_start : i_start + nt, :n_muscles] for i_start in i_starts]
    )
    Xref = (
        np.array(
            [ctrl_data.Xref_fsnn[i_start : i_start + nt, :] for i_start in i_starts]
        )
        if ctrl_data.Xref_fsnn is not None
        else None
    )
    ref_conds = ctrl_data.conditions
    # start at +1 since we floored the start index
    # (trials can start any 2 ms step, while Cleo is at 10 ms steps)
    Ztrials = np.array(
        [ctrl_data.Z[i_start : i_start + nt, :n_muscles] for i_start in i_starts]
    )
    Xtrials = np.array([ctrl_data.X[i_start : i_start + nt, :] for i_start in i_starts])
    assert Ztrials.shape == (ctrl_data.n_trials, nt, n_muscles)

    def acc_pred_cond(refs, true_conds, outputs, n_neighbors=3):
        # Adapt n_neighbors to available samples per condition
        min_samples_per_cond = np.min(np.bincount(true_conds))
        if min_samples_per_cond == 0:
            # Some conditions have no samples - can't do classification
            return np.nan, np.full(len(outputs), np.nan)
        effective_n_neighbors = min(n_neighbors, min_samples_per_cond)
        if effective_n_neighbors < n_neighbors:
            import logging
            logging.warning(
                f"Reduced n_neighbors from {n_neighbors} to {effective_n_neighbors} "
                f"due to limited samples per condition: {np.bincount(true_conds)}"
            )
        classifier = KNeighborsClassifier(n_neighbors=effective_n_neighbors)
        classifier.fit(
            X=refs.reshape((len(refs), -1)),
            y=true_conds,
        )
        acc = classifier.score(
            X=outputs.reshape((len(outputs), -1)),
            y=true_conds,
        )
        return acc, classifier.predict(outputs.reshape((len(outputs), -1)))

    if Xref is not None:
        acc_X, pred_conds_X = acc_pred_cond(Xref, ref_conds, Xtrials)
        acc_X
    acc_Z, pred_conds_Z = acc_pred_cond(Zref, ref_conds, Ztrials)
    results["acc_pred_cond_X"] = acc_X if Xref is not None else np.nan
    results["acc_pred_cond_Z"] = acc_Z

    results["ctrl_compute_latency_mean_ms"] = ctrl_data.compute_latency_mean_ms
    results["ctrl_compute_latency_std_ms"] = ctrl_data.compute_latency_std_ms
    results["ctrl_compute_latency_nonsolve_ms"] = ctrl_data.compute_latency_nonsolve_ms
    results["ctrl_total_latency_ms"] = ctrl_data.total_latency_ms

    results |= pd.read_csv(fsnn_data_analysis_file).iloc[0].to_dict()
    results |= pd.read_csv(model_fit_analysis_file).iloc[0].to_dict()
    print(results)

    df = pd.DataFrame([results])
    print(f"Saving analysis results to {outfile}")
    df.to_csv(outfile, index=False)


if __name__ == "__main__" and "snakemake" in locals():
    fh = setup_logging(snakemake.config, snakemake.log[0])
    setup_debugger(snakemake.config["debug"])

    ctrl_data = CtrlData.load(snakemake.input.ctrl_data)
    null_ctrl_data = CtrlData.load(snakemake.input.null_ctrl_data)
    analyze_ctrl_data(
        snakemake.wildcards,
        preprocess_params(snakemake.config),
        ctrl_data,
        null_ctrl_data,
        snakemake.input.fsnn_data_analysis,
        snakemake.input.model_fit_analysis,
        snakemake.output[0],
    )
