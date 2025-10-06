import importlib
import random
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

from src.data_classes import BaseDataClass
from src.helpers import str2bool
from src.model_fit import LDS
from src.scripts.sys_id_expts import ExptData


@dataclass
class ModelFitPredData(BaseDataClass):
    Xhat_lds: np.ndarray
    Xhat_lds2fsnn: np.ndarray
    Zhat: np.ndarray
    Yhat: np.ndarray
    t_ms: np.ndarray


def model_fit(
    cond_params, base_params, expt_data_file, method_module_file, outfiles, config
):
    extra_seed_int = sum(ord(c) for c in "model_fit")
    seed = int(cond_params["seed"]) + extra_seed_int
    print(f"Seeding with {seed}")
    random.seed(seed)
    np.random.seed(seed)
    # rng = np.random.default_rng(seed)

    expt_data = ExptData.load(expt_data_file)
    n_trials = int(cond_params.n_trials_sys_id)
    assert n_trials <= expt_data.n_trials, f"{n_trials=} > {expt_data.n_trials=}"
    # cut data to end of last trial
    T = expt_data.i_trial_boundaries[n_trials - 1, -1]
    X = expt_data.X[:T]
    Y = expt_data.spikes[:T]
    Z = expt_data.Z[:T]
    if str2bool(cond_params.model_task_inputs):
        U = np.hstack([expt_data.Uopto, expt_data.Utask])
    else:
        U = expt_data.Uopto
    U = U[:T]

    n_x = int(cond_params["n_pcs"])  # just assume we know this

    # Normalize data (removing the mean is essential before training):
    y_mean, y_std = np.mean(Y, axis=0), np.std(Y, axis=0)
    # ensure no zero stds
    y_std[y_std == 0] = 1.0
    # YtrainN, YtestN = (Ytrain - yMean) / yStd, (Ytest - yMean) / yStd
    YN = (Y - y_mean) / y_std
    assert np.all(np.isnan(YN) == False)
    z_mean, z_std = np.mean(Z, axis=0), np.std(Z, axis=0)
    # ZtrainN, ZtestN = (Ztrain - zMean) / zStd, (Ztest - zMean) / zStd
    ZN = (Z - z_mean) / z_std
    u_mean, u_std = np.mean(U, axis=0), np.std(U, axis=0)
    # UtrainN, UtestN = (Utrain - uMean) / uStd, (Utest - uMean) / uStd
    UN = (U - u_mean) / u_std

    lds, Zhat, Yhat, Xhat = importlib.import_module(
        method_module_file.replace(".py", "").replace("/", ".")
    ).fit_lds_to_data(YN, ZN, UN, X, n_x, config, cond_params, base_params, outfiles)

    # saving preprocessing info in model objects to use during inference in runPredict function
    (
        lds.y_mean,
        lds.y_std,
        lds.z_mean,
        lds.z_std,
        lds.u_mean,
        lds.u_std,
    ) = y_mean, y_std, z_mean, z_std, u_mean, u_std

    assert len(X) == len(Xhat)

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
    print(f"Saving predictions to {outfiles.pred_data}")
    model_fit_pred_data.save(outfiles.pred_data)

    print(f"Saving model to {outfiles.lds_model}")
    lds.save(outfiles.lds_model)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    model_fit(
        snakemake.wildcards,
        snakemake.params,
        snakemake.input.sys_id_data,
        snakemake.input.method,
        snakemake.output,
        snakemake.config,
    )
