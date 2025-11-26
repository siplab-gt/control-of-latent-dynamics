import importlib
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

from src.data_classes import BaseDataClass
from src.helpers import preprocess_params, setup_debugger, setup_logging, str2bool
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
    # cut data to end of last trial. wait: why?
    # T = expt_data.i_trial_boundaries[n_trials - 1, -1]
    T = len(expt_data.t_ms)
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

    # Normalize data (removing the mean is essential before training):
    y_mean, y_std = np.mean(Y, axis=0), np.std(Y, axis=0)
    # ensure no zero stds
    y_std[y_std == 0] = 1.0
    # YtrainN, YtestN = (Ytrain - yMean) / yStd, (Ytest - yMean) / yStd
    YN = (Y - y_mean) / y_std
    assert np.all(np.isfinite(YN))
    z_mean, z_std = np.mean(Z, axis=0), np.std(Z, axis=0)
    # ZtrainN, ZtestN = (Ztrain - zMean) / zStd, (Ztest - zMean) / zStd
    ZN = (Z - z_mean) / z_std
    # remember U is normalized, will need to be unnormalized for constraints/costs
    u_mean, u_std = np.mean(U, axis=0), np.std(U, axis=0)
    # UtrainN, UtestN = (Utrain - uMean) / uStd, (Utest - uMean) / uStd
    UN = (U - u_mean) / u_std
    # normalize X too so origin reflects u_mean, y_mean, z_mean inputs
    x_mean, x_std = np.mean(X, axis=0), np.std(X, axis=0)
    XN = (X - x_mean) / x_std

    lds, Zhat, Yhat, Xhat = importlib.import_module(
        method_module_file.replace(".py", "").replace("/", ".")
    ).fit_lds_to_data(YN, ZN, UN, XN, n_x, cond_params, base_params, outfiles, config)

    # saving preprocessing info in model objects to use during inference in runPredict function
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
    lds.save(outfiles.lds_model)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    log_fh = setup_logging(snakemake.config, snakemake.log[0])
    setup_debugger(snakemake.config["debug"])

    model_fit(
        preprocess_params(snakemake.wildcards),
        preprocess_params(snakemake.params),
        snakemake.input.sys_id_data,
        snakemake.input.method,
        snakemake.output,
        preprocess_params(snakemake.config),
    )
