import importlib
import logging
import os
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

from src.data_classes import BaseDataClass
from src.helpers import (
    preprocess_params,
    safe_std_compute,
    sanitize_normalized,
    save_pkl_gz,
    setup_debugger,
    setup_logging,
    str2bool,
)
from src.log_redirector import redirect_output_to_file
from src.model_fit import LDS
from src.scripts.sys_id_expts import ExptData

log = logging.getLogger(__name__)


@dataclass
class ModelFitPredData(BaseDataClass):
    Xhat_lds: np.ndarray
    Xhat_lds2fsnn: np.ndarray
    Zhat: np.ndarray
    Yhat: np.ndarray
    t_ms: np.ndarray


def is_flat(arr, axis=0, thr=0):
    """Returns a bool array that is True for dimensions of axis=0 val that are flat

    Args:
        val (np.array): data
        axis (float, optional): axis over which flatness is evaluated. Defaults to 0.
        thr (float, optional): Threshold for determining flatness. Defaults to 0.

    Returns:
        np.array of bool: array of result for whether each dimension was flat or not
    """
    return (np.max(arr, axis=axis) - np.min(arr, axis=axis)) <= thr


def _deep_merge_overrides(base: dict, override: dict) -> dict:
    """Deep merge `override` into `base` (override wins on conflicts).

    Recurses into nested dicts so a leaf override (e.g.
    usid_hyperparams.loss.loss_weights.forced) replaces only that leaf and
    keeps sibling keys. Mirrors src/model_fit/usid.py::_deep_merge — kept as a
    small local copy so model_fit.py has no cross-module import for a 6-line
    helper (and no import-fallback). Used to fold the loss-weight wildcard
    values into config["usid_overrides"] without clobbering a per-setting
    usid_overrides block.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_overrides(result[key], value)
        else:
            result[key] = value
    return result


def model_fit(
    cond_params, base_params, expt_data_file, method_module_file, outfiles, config
):
    expt_data = ExptData.load(expt_data_file)
    n_trials = int(cond_params.n_trials_sys_id)
    assert n_trials <= expt_data.n_trials, f"{n_trials=} > {expt_data.n_trials=}"
    # cut data to the beginning of the n+1th trial
    try:
        T = expt_data.i_trial_boundaries[n_trials, 0]
    except IndexError:
        T = len(expt_data.X)
    X = expt_data.X[:T]
    Y = expt_data.Y[:T]
    Z = expt_data.Z[:T]
    if str2bool(cond_params.model_task_inputs):
        U = np.hstack([expt_data.Uopto, expt_data.Utask])
    else:
        U = expt_data.Uopto
    U = U[:T]

    for arr in ("Y", "Z", "U"):
        if is_flat(locals()[arr], axis=0).any():
            log.warning(
                f"Warning: some dimensions of {arr} are flat. This may cause issues with model fitting."
            )
            log.warning(
                f"Flat dimensions of {arr}: {np.where(is_flat(locals()[arr], axis=0))[0]}"
            )
            # actually just throw error. this shouldn't happen
            raise ValueError(
                f"Some dimensions of {arr} are flat; there must be something wrong with the data."
            )

    n_x = X.shape[1]  # just assume we know this

    # Normalize data (mean+std with channel-degenerate guard, then post-cast
    # overflow sanitizer). Uses shared helpers from src.helpers so the same
    # logic runs on the natural-electrode normalization in
    # src/model_fit/usid.py::_load_and_normalize_natural_data.
    #
    # Clip counts are accumulated into sanitize_clip_counts and stashed on
    # the fitted lds so analyze_model_fit can surface them as summary.csv
    # columns (n_sanitize_clipped_*). A non-zero count on any channel means
    # upstream pathology — typically v3 plant divergence at extreme k/m —
    # and the row should be filtered from scientific comparisons.
    sanitize_clip_counts = {}
    def _normalize(arr, name):
        m, s = safe_std_compute(arr, axis=0, name=name, log=log)
        normalized, n_clipped = sanitize_normalized((arr - m) / s, name, log=log)
        sanitize_clip_counts[name] = n_clipped
        return m, s, normalized

    y_mean, y_std, YN = _normalize(Y, "Y")
    assert np.all(np.isfinite(YN))
    z_mean, z_std, ZN = _normalize(Z, "Z")
    # remember U is normalized, will need to be unnormalized for constraints/costs
    u_mean, u_std, UN = _normalize(U, "U")
    # normalize X too so origin reflects u_mean, y_mean, z_mean inputs
    x_mean, x_std, XN = _normalize(X, "X")

    # Make normalization stats available to fit methods (used by USID inline training)
    config['y_mean'] = y_mean
    config['y_std'] = y_std
    config['u_mean'] = u_mean
    config['u_std'] = u_std
    # z_mean/z_std let usid.py normalize the natural behavior (natural_expt.Z) into the
    # controller's Z coordinates for the Phase-2 B1'/Cz1 pretrain (learnable_cz path).
    config['z_mean'] = z_mean
    config['z_std'] = z_std

    # Opsin-fit validation data (only present when sys_id_expts recorded it).
    # Sliced to the same T window used for YN/UN/ZN so downstream diagnostics
    # stay aligned with the training/validation split.
    if expt_data.Iopto_per_neuron is not None:
        config['Iopto_per_neuron'] = expt_data.Iopto_per_neuron[:T]
        config['channel_neuron_transmittance'] = expt_data.channel_neuron_transmittance

    lds, Zhat, Yhat, Xhat = importlib.import_module(
        method_module_file.replace(".py", "").replace("/", ".")
    ).fit_lds_to_data(YN, ZN, UN, XN, n_x, cond_params, base_params, outfiles, config)

    # saving preprocessing info in model objects to use during inference in runPredict function
    # Note: This overwrites any stats set by the fit function (e.g., usid.py, braid.py).
    # The fit functions receive pre-normalized data, so we store the original stats here
    # for proper denormalization at inference time.
    (
        lds.y_mean,
        lds.y_std,
        lds.z_mean,
        lds.z_std,
        lds.u_mean,
        lds.u_std,
        lds.x_mean,
        lds.x_std,
    ) = y_mean, y_std, z_mean, z_std, u_mean, u_std, x_mean, x_std

    # Merge sys_id normalization clip counts with any counts the fit method
    # accumulated (e.g., usid.py adds "Natural_YN" after loading natural
    # electrode data). Dict union; fit-method keys don't collide with Y/Z/U/X.
    fit_counts = getattr(lds, "sanitize_clip_counts", {}) or {}
    lds.sanitize_clip_counts = {**sanitize_clip_counts, **fit_counts}

    assert len(XN) == len(Xhat)

    # linear regression to map
    lds.linreg_lds2fsnn = LinearRegression().fit(Xhat, X)
    Xhat_lds2fsnn = lds.linreg_lds2fsnn.predict(Xhat)
    t_ms = expt_data.t_ms[: len(Xhat_lds2fsnn)]
    model_fit_pred_data = ModelFitPredData(
        Xhat_lds=Xhat,
        Xhat_lds2fsnn=Xhat_lds2fsnn,
        Zhat=Zhat * z_std + z_mean,
        Yhat=Yhat * y_std + y_mean,
        t_ms=t_ms,
    )
    log.info(f"Saving predictions to {outfiles.pred_data}")
    model_fit_pred_data.save(outfiles.pred_data)

    log.info(f"Saving model to {outfiles.lds_model}")
    save_pkl_gz(lds, outfiles.lds_model)

    # Write fit-quality analysis atomically with the LDS so the two can't
    # go out of sync. Keeps the CSV fresh whenever model_fit retrains.
    from src.scripts.analyze_model_fit import analyze_model_fit
    log.info(f"Writing fit-quality analysis to {outfiles.analysis_csv}")
    analyze_model_fit(
        cond_params, lds, model_fit_pred_data, expt_data, outfiles.analysis_csv,
    )


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    log_fh = setup_logging(snakemake.config, snakemake.log[0])
    setup_debugger(snakemake.config["debug"])

    config_dict = preprocess_params(snakemake.config)
    # Pass natural electrode data path for USID inline training
    natural_data = getattr(snakemake.input, 'natural_data', None)
    if natural_data:
        config_dict['natural_data_path'] = str(natural_data)

    # Per-setting overlay: keys placed in config[f"model_fit_{setting}"]
    # override their top-level counterparts. Lets `setting` sweep choose
    # use_opsin / model_fit_method / etc. without globally toggling them.
    # Logs every clobber at INFO so a future user can see what the active
    # setting silently changed (drift hazard: e.g. use_opsin lives both
    # top-level and per-setting in some configs).
    setting = getattr(snakemake.wildcards, "setting", None)
    if setting:
        overlay = snakemake.config.get(f"model_fit_{setting}", {}) or {}
        for k, v in overlay.items():
            old = config_dict.get(k, "<unset>")
            if old != v:
                log.info(
                    f"[setting={setting}] model_fit overlay: "
                    f"{k}={old!r} -> {v!r}"
                )
            config_dict[k] = v

    # Route the model_fit-level loss-weight wildcards into the USID override
    # stack. These are promoted to sweep wildcards (Snakefile
    # steps_params["model_fit"]) so a sweep over them parallelizes into one
    # model_fit output subtree per combo, rather than needing a sequential
    # config-sed of cold_usid.yaml. _load_usid_config deep-merges
    # config["usid_overrides"] LAST (after cold_usid.yaml + runtime fields),
    # so these leaves win at the usid.py read sites
    # (usid_hyperparams.loss.loss_weights.*).
    #
    # Snakemake wildcards are strings; parse explicitly — every loss-weight
    # leaf is a float ("0.0", "2.0"). Deep-merge so a per-setting
    # usid_overrides block (e.g. the usid_loss_* loss schedules) is preserved.
    wc_forced = getattr(snakemake.wildcards, "forced", None)
    wc_usid_rollout = getattr(snakemake.wildcards, "usid_rollout", None)
    # Phase-2 B1'/Cz1 pretrain (b1_recon) + Phase-4 Cz2/B2' finetune (z_recon) z/y weights,
    # plus the USID teacher loss (which — like rollout — also has a SCHEDULE that must be co-overridden).
    wc_b1_z = getattr(snakemake.wildcards, "b1_z", None)
    wc_b1_y = getattr(snakemake.wildcards, "b1_y", None)
    wc_b2_z = getattr(snakemake.wildcards, "b2_z", None)
    wc_b2_y = getattr(snakemake.wildcards, "b2_y", None)
    # cz2_warm routes z_recon.cz2_warm_start (Phase-4 Cz2 ridge warm-start vs random init).
    wc_cz2_warm = getattr(snakemake.wildcards, "cz2_warm", None)
    wc_usid_teacher = getattr(snakemake.wildcards, "usid_teacher", None)
    # usid_delta routes loss_weights.delta (the delta-of-rollout loss, parallel to
    # the rollout level loss). It has NO schedule entry, so it is a plain leaf write
    # like forced — no schedule co-override needed. Held at 0 to isolate the rollout
    # axis when usid_rollout is swept.
    wc_usid_delta = getattr(snakemake.wildcards, "usid_delta", None)
    # n_restarts routes usid_hyperparams.n_restarts (top-level, NOT a loss leaf): the
    # base-fit multistart count; >1 fits N base models and keeps the best forced-path R2(Y).
    wc_n_restarts = getattr(snakemake.wildcards, "n_restarts", None)
    if any(w is not None for w in (wc_forced, wc_usid_rollout, wc_b1_z, wc_b1_y,
                                   wc_b2_z, wc_b2_y, wc_usid_teacher, wc_usid_delta,
                                   wc_n_restarts, wc_cz2_warm)):
        wildcard_usid_overrides = {"usid_hyperparams": {}}
        # forced + usid_rollout + usid_teacher share the loss.{loss_weights,schedule} leaves.
        loss_weights = {}
        if wc_forced is not None:
            loss_weights["forced"] = float(wc_forced)
        if wc_usid_rollout is not None:
            loss_weights["rollout"] = float(wc_usid_rollout)
        if wc_usid_teacher is not None:
            loss_weights["teacher"] = float(wc_usid_teacher)
        if wc_usid_delta is not None:
            loss_weights["delta"] = float(wc_usid_delta)
        loss_block = {}
        if loss_weights:
            loss_block["loss_weights"] = loss_weights
        # cold_usid.yaml pins rollout AND teacher via USID-phase SCHEDULEs (linear ramps that persist
        # past their window), so setting loss_weights.{rollout,teacher} alone is silently overridden.
        # Co-override the schedule to the swept value (flat start=end=v) so a 0 stays 0 with no ramp.
        schedule = {}
        if wc_usid_rollout is not None:
            r = float(wc_usid_rollout)
            schedule["rollout"] = {"mode": "linear", "start": r, "end": r, "epochs": 20}
        if wc_usid_teacher is not None:
            t = float(wc_usid_teacher)
            schedule["teacher"] = {"mode": "linear", "start": t, "end": t, "epochs": 20}
        if schedule:
            loss_block["schedule"] = schedule
        # Phase-2 B1'/Cz1 pretrain (b1_recon) + Phase-4 Cz2/B2' finetune (z_recon) z/y weight overrides —
        # plain leaf writes (these blocks have no schedule). Swept independently here
        # (default b1 ↔ b2 equal in cold_usid).
        b1_recon = {}
        if wc_b1_z is not None:
            b1_recon["z_weight"] = float(wc_b1_z)
        if wc_b1_y is not None:
            b1_recon["y_weight"] = float(wc_b1_y)
        if b1_recon:
            loss_block["b1_recon"] = b1_recon
        z_recon = {}
        if wc_b2_z is not None:
            z_recon["z_weight"] = float(wc_b2_z)
        if wc_b2_y is not None:
            z_recon["y_weight"] = float(wc_b2_y)
        if wc_cz2_warm is not None:
            z_recon["cz2_warm_start"] = (wc_cz2_warm == "ridge")
        if z_recon:
            loss_block["z_recon"] = z_recon
        if loss_block:
            wildcard_usid_overrides["usid_hyperparams"]["loss"] = loss_block
        if wc_n_restarts is not None:
            wildcard_usid_overrides["usid_hyperparams"]["n_restarts"] = int(wc_n_restarts)
        existing = config_dict.get("usid_overrides", {}) or {}
        config_dict["usid_overrides"] = _deep_merge_overrides(
            existing, wildcard_usid_overrides
        )
        log.info(
            f"[model_fit] wildcard usid_overrides: forced={wc_forced!r}, "
            f"usid_rollout={wc_usid_rollout!r}, b1_z={wc_b1_z!r}, "
            f"b1_y={wc_b1_y!r}, b2_z={wc_b2_z!r}, b2_y={wc_b2_y!r}, "
            f"usid_teacher={wc_usid_teacher!r}, usid_delta={wc_usid_delta!r}, "
            f"n_restarts={wc_n_restarts!r}, cz2_warm={wc_cz2_warm!r}"
        )

    model_fit(
        preprocess_params(snakemake.wildcards),
        preprocess_params(snakemake.params),
        snakemake.input.sys_id_data,
        snakemake.input.method,
        snakemake.output,
        config_dict,
    )
