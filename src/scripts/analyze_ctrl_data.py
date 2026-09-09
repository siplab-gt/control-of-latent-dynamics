"""Analyze control experiment data and include fSNN and model fit analyses."""

import json

import brian2.only as b2
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from snakemake.script import snakemake

from src.helpers import (
    drop_constant_behavior,
    load_factors_emg,
    load_pkl_gz,
    preprocess_params,
    seed_everything,
    setup_debugger,
    setup_logging,
)
from src.scripts.attribution_metrics import compute_attribution_metrics
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


def _curve_gain_retention(x, y, eval_points):
    """Mean fractional small-signal gain retained at `eval_points` on a monotone
    response curve y(x): mean_i y'(eval_i) / y'(x_min). Shared by the
    commanded-light operating-regime metric (bench curve, x[0]=0 so y'(x_min)=f'(0))
    and the per-neuron metric (empirical activation curve, x_min = lowest-drive
    bin). Returns NaN on a degenerate curve or no eval points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    eval_points = np.asarray(eval_points, dtype=float)
    if x.size < 2 or eval_points.size == 0:
        return np.nan
    order = np.argsort(x)
    x, y = x[order], y[order]
    slope = np.gradient(y, x)
    slope0 = float(slope[0])
    if not np.isfinite(slope0) or slope0 <= 0:
        return np.nan
    return float(np.mean(np.interp(eval_points, x, slope)) / slope0)


def _empirical_curve(x, y, nbins=10, min_count=50):
    """Monotone-x empirical response curve: log-spaced bins over x>0, mean y per
    bin. Returns (bin_centers, bin_means) for bins with >= min_count samples.
    Log-spaced because per-neuron received irradiance is heavily right-skewed
    (most neuron-time samples ~0); linear bins collapse at the low end."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pos = x > 0
    if pos.sum() < min_count:
        return np.array([]), np.array([])
    lo, hi = float(x[pos].min()), float(x[pos].max())
    if hi <= lo:
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    centers, means = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x < b)
        if sel.sum() >= min_count:
            centers.append(float(x[sel].mean()))
            means.append(float(y[sel].mean()))
    return np.array(centers), np.array(means)


def per_neuron_saturation(ctrl_data, eval_mask, E_rev, v_rest, active_floor=1e-3):
    """Realized per-neuron opsin saturation during control — the faithful
    complement to the commanded-light G_op. It uses the activation the neurons
    ACTUALLY experience: per-neuron received irradiance (Uopto @ transmittance,
    so it accounts for tissue attenuation + multi-fiber overlap) and the recorded
    per-neuron photocurrent Iopto_per_neuron, so it captures the (V−E) driving
    force and kinetics that the commanded-light/bench metric cannot.

    Always returns the full column dict (NaN-filled) so the summary schema is
    stable; values are NaN when the per-neuron photocurrent was not recorded
    (control runs predating the ctrl_expts StateMonitor) or there is too little
    illuminated activation. Diagnostic — the caller wraps it in try/except.

    Columns (over eval-window, illuminated neuron-time samples):
    - ctrl_perneuron_gain_retention: realized incremental gain at the operating
      drive vs small-signal, from the empirical (received irradiance -> |Iopto|)
      curve. <1 = saturating; folds gating + (V−E) + kinetics together.
    - ctrl_perneuron_drive_retention (needs V): irradiance-weighted mean
      (V−E)/(v_rest−E) — the fraction of resting driving force remaining.
    - ctrl_perneuron_gating_retention (needs V): gain retention of the
      driving-force-removed photocurrent |Iopto|/|V−E| vs received irradiance —
      the light-gating part (what phim could move).
    - ctrl_perneuron_frac_saturated: fraction of illuminated samples whose local
      realized gain has dropped below half the small-signal gain.
    - ctrl_perneuron_light_max / n_active_neurons: operating-point spread.

    All retentions are count-weighted over illuminated neuron-time samples (the
    low-drive bulk dominates the mean; read with frac_saturated for the saturated
    stim-carrying tail). gain ≈ gating × drive holds only approximately (gain and
    gating are slope-ratio retentions of empirical curves; drive is a level-ratio
    mean), so treat the split as indicative, not an exact factorization.
    """
    out = {
        "ctrl_perneuron_gain_retention": np.nan,
        "ctrl_perneuron_drive_retention": np.nan,
        "ctrl_perneuron_gating_retention": np.nan,
        "ctrl_perneuron_frac_saturated": np.nan,
        "ctrl_perneuron_light_max": np.nan,
        "ctrl_perneuron_n_active_neurons": 0,
    }
    I = ctrl_data.Iopto_per_neuron
    Tn = ctrl_data.channel_neuron_transmittance
    if I is None or Tn is None:
        return out
    I = np.abs(np.asarray(I, dtype=float))
    E = np.asarray(ctrl_data.Uopto, dtype=float) @ np.asarray(Tn, dtype=float)  # (T, N) received irradiance
    out["ctrl_perneuron_light_max"] = float(E.max()) if E.size else np.nan
    out["ctrl_perneuron_n_active_neurons"] = int(np.sum(E.mean(0) > 0)) if E.size else 0

    e = E[eval_mask].ravel()
    i = I[eval_mask].ravel()
    thr = max(1e-9, active_floor * (float(e.max()) if e.size else 0.0))
    act = e > thr
    if act.sum() < 200:
        return out
    ea, ia = e[act], i[act]

    centers, resp = _empirical_curve(ea, ia)
    out["ctrl_perneuron_gain_retention"] = _curve_gain_retention(centers, resp, ea)
    if centers.size >= 2:
        slope = np.gradient(resp, centers)
        s0 = float(slope[0])
        if np.isfinite(s0) and s0 > 0:
            local = np.interp(ea, centers, slope) / s0
            out["ctrl_perneuron_frac_saturated"] = float(np.mean(local < 0.5))

    V = ctrl_data.V_per_neuron
    if V is not None and np.isfinite(E_rev) and np.isfinite(v_rest):
        dV = np.asarray(V, dtype=float)[eval_mask].ravel()[act] - E_rev
        dV0 = v_rest - E_rev
        if dV0 != 0:
            # count-weighted over illuminated samples, matching the gain-retention
            # convention (G_op / _curve_gain_retention); frac_saturated carries
            # the high-drive (stim-carrying neuron) tail.
            out["ctrl_perneuron_drive_retention"] = float(np.mean(dV / dV0))
            valid = np.abs(dV) > 1e-9
            if valid.sum() > 200:
                g_raw = ia[valid] / np.abs(dV[valid])  # driving-force-removed photocurrent (= gating x rho)
                c2, r2 = _empirical_curve(ea[valid], g_raw)
                out["ctrl_perneuron_gating_retention"] = _curve_gain_retention(c2, r2, ea[valid])
    return out


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
    if u_active.size == 0:
        return out

    # G_op = mean f'(u)/f'(0) over realized commands on the bench gating curve
    # (shared kernel; the bench grid starts at 0 so f'(x_min) == f'(0)).
    out["ctrl_opsin_gain_retention"] = _curve_gain_retention(light, ss, u_active)

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
    lds_model_file,
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

    # Zref comes back from the controller already spanning the model's behavior channels, while
    # ctrl_data.Z is the raw recorded readout; trim Z (and the NULL twin) to match.
    ctrl_Z = drop_constant_behavior(ctrl_data.Z, "ctrl Z")
    null_Z = drop_constant_behavior(null_ctrl_data.Z, "null ctrl Z")

    # R2
    results["R2_track_Zref_Z"] = r2_score_masked(ctrl_data.Zref, ctrl_Z)
    results["R2_track_Xref_X_fsnn"] = r2_score_masked(ctrl_data.Xref_fsnn, ctrl_data.X)
    results["R2_track_Xref_Xhat_fsnn"] = r2_score_masked(
        ctrl_data.Xref_fsnn, ctrl_data.Xhat_fsnn
    )
    results["R2_track_Xref_Xhat_lds"] = r2_score_masked(
        ctrl_data.Xref_lds, ctrl_data.Xhat_lds
    )

    # Q
    results["Q_track_Zref_Z"] = q_track_masked(
        ctrl_data.Zref, ctrl_Z, null_Z
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
        # Realized per-neuron saturation (faithful complement; NaN-filled when
        # the per-neuron photocurrent wasn't recorded for this cell).
        results |= per_neuron_saturation(
            ctrl_data, eval_mask,
            E_rev=float(curve.get("E_rev", np.nan)),
            v_rest=float(curve.get("v_rest", np.nan)),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WARN] operating-regime linearity metric failed: {e}")

    # Attribution / routing / counterfactual metrics. Diagnostics-class:
    # a failure here must not cost the cell its tracking metrics, which are the primary output.
    try:
        results.update(compute_attribution_metrics(
            ctrl_data,
            null_ctrl_data,
            load_pkl_gz(lds_model_file),
            eval_mask,
            np.random.default_rng(seed_everything(int(cond_params["seed"]), "attribution")),
        ))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WARN] attribution metrics failed: {e}")

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
        snakemake.input.lds_model,
        snakemake.output[0],
    )
