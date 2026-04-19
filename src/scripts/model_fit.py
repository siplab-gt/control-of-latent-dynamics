import importlib
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

from src.data_classes import BaseDataClass
from src.helpers import preprocess_params, save_pkl_gz, setup_debugger, setup_logging, str2bool
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

    def _safe_std(arr: np.ndarray, axis: int, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean + std with defensive guards against degenerate channels.

        Historically only the ``std == 0`` case was guarded (constant channel).
        Under the v3 chr2_4s_fsnn plant some LFP channels can carry values
        large enough that ``np.std`` overflows (squared deviations exceed
        float range), leaving an ``inf`` or ``nan`` in ``std``. That inf then
        silently contaminates ``y_std`` → corrupts the de-normalization step
        inside USID diagnostics + analyze_model_fit → NaN Yhat → sklearn
        rejects → crash. Clamp every non-finite / non-positive std to 1.0
        and warn loudly so the upstream extreme-value signal isn't lost.
        """
        m = np.mean(arr, axis=axis)
        s = np.std(arr, axis=axis)
        bad = ~np.isfinite(s) | (s <= 0)
        if bad.any():
            n_bad = int(bad.sum())
            arr_max = float(np.max(np.abs(arr[np.isfinite(arr)]))) if np.any(np.isfinite(arr)) else float("nan")
            log.warning(
                f"{name}: {n_bad} channel(s) have non-finite or zero std (likely "
                f"overflow from extreme values; max |{name}|={arr_max:g}). Clamping "
                f"their std to 1.0 so de-normalization stays finite. Check upstream "
                f"data for pathological magnitudes."
            )
            s = np.where(bad, 1.0, s)
        return m, s

    # Normalize data (removing the mean is essential before training):
    y_mean, y_std = _safe_std(Y, axis=0, name="Y")
    YN = (Y - y_mean) / y_std
    assert np.all(np.isfinite(YN))
    z_mean, z_std = _safe_std(Z, axis=0, name="Z")
    ZN = (Z - z_mean) / z_std
    # remember U is normalized, will need to be unnormalized for constraints/costs
    u_mean, u_std = _safe_std(U, axis=0, name="U")
    UN = (U - u_mean) / u_std
    # normalize X too so origin reflects u_mean, y_mean, z_mean inputs
    x_mean, x_std = _safe_std(X, axis=0, name="X")
    XN = (X - x_mean) / x_std

    # Make normalization stats available to fit methods (used by USID inline training)
    config['y_mean'] = y_mean
    config['y_std'] = y_std
    config['u_mean'] = u_mean
    config['u_std'] = u_std

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

    model_fit(
        preprocess_params(snakemake.wildcards),
        preprocess_params(snakemake.params),
        snakemake.input.sys_id_data,
        snakemake.input.method,
        snakemake.output,
        config_dict,
    )
