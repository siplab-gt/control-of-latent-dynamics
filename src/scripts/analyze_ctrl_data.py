"""Analyze control experiment data and include fSNN and model fit analyses."""

import json

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


def operating_regime_linearity(ctrl_data, eval_mask, curve, u_max):
    """How linear is the portion of the opsin saturation curve the controller
    actually drives, given its realized commanded light `ctrl_data.Uopto`.

    `curve` is the dict written by the `opsin_saturation_curve` diagnostic: a
    light grid (mW/mm2) and the steady-state |Iopto| response on it, i.e. the
    static actuator map f(u). This complements the fixed-range `lin[5,75]`
    (a property of f alone) by weighting it by where the controller operates.

    - `ctrl_opsin_gain_retention` (G_op) = mean_t f'(u_t) / f'(0): the average
      fractional small-signal gain the opsin still delivers at the commanded
      light levels. 1.0 = the controller stays in the linear region; -> 0 = it
      drives the opsin deep into saturation.
    - `ctrl_opsin_lin_op` = f(u_hi) / ((u_hi/u_lo) f(u_lo)) over the realized
      operating range [p10, p90] of commanded light: the same secant-linearity
      form as lin[5,75] but anchored at what the controller actually commands.
      Read it alongside `ctrl_opsin_gain_retention` and the p10/p90 columns: a
      narrow operating band (p10 ~ p90) drives lin_op -> 1 regardless of where
      on the curve it sits, so gain_retention is the disambiguating number.
    - operating-point stats: mean / max commanded light and the fraction of
      commands at the u_max bound.

    Diagnostic only — the caller wraps this in try/except so a failure never
    blocks the summary row. Commanded light is in mW/mm2 (source irradiance),
    matching the curve grid and u_max. Restricted to the eval windows for
    consistency with Q_track/R2_track; zero-light commands are excluded from
    the gain statistics (they carry no operating-point information).
    """
    U = np.asarray(ctrl_data.Uopto, dtype=float)
    light = np.asarray(curve["light_grid_mWmm2"], dtype=float)
    ss = np.abs(np.asarray(curve["iopto_ss_abs"], dtype=float))
    order = np.argsort(light)
    light, ss = light[order], ss[order]
    slope = np.gradient(ss, light)
    # Small-signal gain f'(0): the near-origin forward-difference slope of the
    # curve (light[0] is 0). Used only to normalize G_op, so G_op == 1 exactly
    # for a linear curve and < 1 once the realized commands enter the bend.
    slope0 = float(np.interp(0.0, light, slope))

    u_all = U[eval_mask].ravel()
    u_active = u_all[u_all > 0.0]
    out = {
        "ctrl_opsin_light_mean": float(np.mean(u_all)) if u_all.size else np.nan,
        "ctrl_opsin_light_mean_active": float(np.mean(u_active)) if u_active.size else np.nan,
        "ctrl_opsin_light_max": float(np.max(u_all)) if u_all.size else np.nan,
        "ctrl_opsin_frac_light_at_umax": (
            float(np.mean(u_all >= 0.99 * u_max)) if (u_all.size and u_max > 0) else np.nan
        ),
        "ctrl_opsin_gain_retention": np.nan,
        "ctrl_opsin_lin_op": np.nan,
        "ctrl_opsin_op_light_p10": np.nan,
        "ctrl_opsin_op_light_p90": np.nan,
    }
    if u_active.size == 0 or not np.isfinite(slope0) or slope0 <= 0:
        return out

    fprime = np.interp(u_active, light, slope)
    out["ctrl_opsin_gain_retention"] = float(np.mean(fprime) / slope0)

    u_lo = float(np.percentile(u_active, 10))
    u_hi = float(np.percentile(u_active, 90))
    out["ctrl_opsin_op_light_p10"] = u_lo
    out["ctrl_opsin_op_light_p90"] = u_hi
    f_lo = float(np.interp(u_lo, light, ss))
    f_hi = float(np.interp(u_hi, light, ss))
    if u_lo > 0 and f_lo > 0:
        out["ctrl_opsin_lin_op"] = f_hi / ((u_hi / u_lo) * f_lo)
    return out


def analyze_ctrl_data(
    cond_params,
    cfg,
    ctrl_data: CtrlData,
    null_ctrl_data: CtrlData,
    fsnn_data_analysis_file,
    model_fit_analysis_file,
    chr2_stability_file,
    saturation_curve_file,
    outfile,
):
    results = {**cond_params}
    eval_mask = np.concat(
        [np.arange(start, end) for start, end in ctrl_data.i_trial_boundaries]
    )
    assert np.all(ctrl_data.i_trial_boundaries == null_ctrl_data.i_trial_boundaries)

    # Pathology flags FIRST, before any r2/q computation. If a row has NaN,
    # sklearn's r2_score raises ValueError by default and would kill the
    # whole rule — then summary.csv would be missing the row entirely and
    # the n_nonfinite_* counts would never be written. So: count first,
    # then wrap each metric in try/except → np.nan. Non-zero n_nonfinite_*
    # flags a pathological row; downstream aggregation should filter.
    #
    # NOTE: Yhat/Yref/Zhat/Zref are NOT currently counted — ignored for
    # now. They can also carry NaN under v3 plant divergence, but in
    # practice NaN propagation starts in the state estimates (Xhat/Xref)
    # and leaks outward, so the X counts are the primary signal.
    def _count_nonfinite(arr):
        if arr is None or np.size(arr) == 0:
            return 0
        return int(np.sum(~np.isfinite(np.asarray(arr))))
    results["n_nonfinite_Xhat_lds"] = _count_nonfinite(ctrl_data.Xhat_lds)
    results["n_nonfinite_Xref_lds"] = _count_nonfinite(ctrl_data.Xref_lds)
    results["n_nonfinite_Xhat_fsnn"] = _count_nonfinite(ctrl_data.Xhat_fsnn)
    results["n_nonfinite_Xref_fsnn"] = _count_nonfinite(ctrl_data.Xref_fsnn)

    def r2_score_masked(y_true, y_pred):
        if is_empty(y_true) or is_empty(y_pred):
            return np.nan
        try:
            # multioutput="raw_values" + force_finite=False so per-dim R²
            # comes back as -Inf when the true series is constant
            # (zero-variance denominator). The reaching task has a
            # constant 3rd output dim (`output_2`) — under sklearn's
            # default `force_finite=True` + `multioutput="uniform_average"`,
            # that dim's R² collapses to 0.0 or 1.0 depending on prediction
            # match and silently shifts the average. Mirroring
            # `analyze_fsnn_data.py`'s convention (mean-over-finite)
            # excludes the constant dim cleanly.
            per_dim = r2_score(
                y_true[eval_mask], y_pred[eval_mask],
                multioutput="raw_values", force_finite=False,
            )
            finite = np.isfinite(per_dim)
            if not finite.any():
                return np.nan
            return float(np.mean(per_dim, where=finite))
        except ValueError:
            # sklearn's r2_score rejects NaN/inf by default. Return NaN
            # so the summary row still writes; the n_nonfinite_* flags
            # above mark this row as pathological.
            return np.nan

    def q_track_masked(ref, achieved, baseline):
        if is_empty(ref) or is_empty(achieved) or is_empty(baseline):
            return np.nan
        try:
            r = ref[eval_mask]
            a = achieved[eval_mask]
            b = baseline[eval_mask]
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(a))
                    and np.all(np.isfinite(b))):
                return np.nan
            return 1 - nrmse(r, a, b)
        except (ValueError, ZeroDivisionError):
            return np.nan

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

    # KNN classification — also vulnerable to NaN in Xref / Xtrials via
    # sklearn's check_array. Skip gracefully when pathological so the row
    # still writes.
    if Xref is not None and np.all(np.isfinite(Xref)) and np.all(np.isfinite(Xtrials)):
        try:
            acc_X, pred_conds_X = acc_pred_cond(Xref, ref_conds, Xtrials)
        except ValueError:
            acc_X = np.nan
    else:
        acc_X = np.nan
    if np.all(np.isfinite(Zref)) and np.all(np.isfinite(Ztrials)):
        try:
            acc_Z, pred_conds_Z = acc_pred_cond(Zref, ref_conds, Ztrials)
        except ValueError:
            acc_Z = np.nan
    else:
        acc_Z = np.nan
    results["acc_pred_cond_X"] = acc_X
    results["acc_pred_cond_Z"] = acc_Z

    results["ctrl_compute_latency_mean_ms"] = ctrl_data.compute_latency_mean_ms
    results["ctrl_compute_latency_std_ms"] = ctrl_data.compute_latency_std_ms
    results["ctrl_compute_latency_nonsolve_ms"] = ctrl_data.compute_latency_nonsolve_ms
    results["ctrl_total_latency_ms"] = ctrl_data.total_latency_ms

    results |= pd.read_csv(fsnn_data_analysis_file).iloc[0].to_dict()
    results |= pd.read_csv(model_fit_analysis_file).iloc[0].to_dict()
    results |= pd.read_csv(chr2_stability_file).iloc[0].to_dict()

    # Operating-regime opsin linearity (diagnostic — must never block the row).
    # Scores the controller's realized commanded light against the plant's
    # saturation curve, and carries the plant's static linearity indices
    # through as provenance columns.
    try:
        with open(saturation_curve_file) as f:
            curve = json.load(f)
        u_max = float(cfg.expts.u_max_mWmm2)
        results |= operating_regime_linearity(ctrl_data, eval_mask, curve, u_max)
        for key in (
            "linearity_index_5_to_75",
            "linearity_index_5_to_20",
            "i_half_light_mWmm2",
            "marginal_gain_at_umax",
        ):
            results[f"opsin_{key}"] = curve.get(key, np.nan)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WARN] operating-regime linearity metric failed: {e}")

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
        snakemake.input.chr2_stability,
        snakemake.input.saturation_curve,
        snakemake.output[0],
    )
