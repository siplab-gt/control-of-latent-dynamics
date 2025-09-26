import os
import random
from dataclasses import dataclass

import BRAID
import dill
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from src.helpers import str2bool
from src.plot_style import style_plots
from src.scripts.plot_expt_data import plot_all
from src.scripts.sys_id_expts import ExptData


def model_fit(cond_params, base_params, expt_data_file, outfiles, config):
    use_cpu = config["model_fit_device"].lower() == "cpu"
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Using CPU for model fitting")
    else:
        print("Using GPU for model fitting")
    BRAID.tools.tf_tools.setupTensorflow(cpu=use_cpu)
    import tensorflow as tf
    from BRAID.BRAIDModel import BRAIDModel, runPredict
    from BRAID.MainModel import MainModel, MainModelPrepareArgs

    extra_seed_int = sum(ord(c) for c in "model_fit")
    seed = int(cond_params["seed"]) + extra_seed_int
    print(f"Seeding with {seed}")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # rng = np.random.default_rng(seed)

    k = int(cond_params["k"])

    # numpy instead of ExptData to be able to run in isolated env
    expt_data = ExptData.load(expt_data_file)
    n_trials = int(cond_params.n_trials_sys_id)
    assert n_trials <= expt_data.n_trials
    # cut data to end of last trial
    T = expt_data.i_trial_boundaries[n_trials, -1]
    X = expt_data.X[:T]
    Y = expt_data.spikes[:T]
    Z = expt_data.Z[:T]
    if str2bool(cond_params.model_task_inputs):
        U = np.hstack([expt_data.Uopto, expt_data.Utask])
    else:
        U = expt_data.Uopto
    U = U[:T]

    braid_model: BRAIDModel = BRAIDModel()
    methodCode = (
        "BRAID_Cz1HL64U"  # Only Cz will be nonlinear, 1 Hidden layer with 64 Units.
    )
    args = MainModelPrepareArgs(methodCode)
    args["epochs"] = int(base_params.model_fit["epochs"])
    n_x = int(cond_params["n_pcs"])  # just assume we know this

    # Separate data into training and test data:
    # T_train = T_total // 2
    # Ytrain, Ytest = Y[:T_train], Y[T_train:]
    # Ztrain, Ztest = Z[:T_train], Z[T_train:]
    # Utrain, Utest = U[:T_train], U[T_train:]
    # actually, just use built-in validation set

    # Normalize data (removing the mean is essential before training):
    yMean, yStd = np.mean(Y, axis=0), np.std(Y, axis=0)
    # ensure no zero stds
    yStd[yStd == 0] = 1.0
    # YtrainN, YtestN = (Ytrain - yMean) / yStd, (Ytest - yMean) / yStd
    YN = (Y - yMean) / yStd
    assert np.all(np.isnan(YN) == False)
    zMean, zStd = np.mean(Z, axis=0), np.std(Z, axis=0)
    # ZtrainN, ZtestN = (Ztrain - zMean) / zStd, (Ztest - zMean) / zStd
    ZN = (Z - zMean) / zStd
    uMean, uStd = np.mean(U, axis=0), np.std(U, axis=0)
    # UtrainN, UtestN = (Utrain - uMean) / uStd, (Utest - uMean) / uStd
    UN = (U - uMean) / uStd

    braid_model.fit(
        YN.T,
        ZN.T,
        U=UN.T,
        nx=n_x,
        n1=n_x,  # I think we _only_ want behaviorally relevant factors
        n3=None,  # by default
        n_pre=None,
        args_base=args,
        create_val_from_training=True,
    )
    # saving preprocessing info in model objects to use during inference in runPredict function
    (
        braid_model.yMean,
        braid_model.yStd,
        braid_model.zMean,
        braid_model.zStd,
        braid_model.uMean,
        braid_model.uStd,
    ) = yMean, yStd, zMean, zStd, uMean, uStd

    assert braid_model.n1 == braid_model.nx == n_x

    # estimate Q, R
    zPred, yPred, xPred, _, _, _ = runPredict(
        braid_model, Y, Z, U, YType="cont", ZType="cont"
    )
    # runPredict function outputs are lists of length= #steps_ahead desired for inference which is taken from the field: idSysF.step_ahead
    # Here we are only looking at 1-step-ahead predictions i.e., the lists only have one element:
    zPred, yPred, xPred = zPred[0], yPred[0], xPred[0]
    print(f"{zPred.shape=}, {yPred.shape=}, {xPred.shape=}")
    lssm = braid_model.getLSSM()
    # problem with yPred. Will recalculate instead of debugging
    # yPred = xPred @ lssm.C.T  # should be normalized
    yres = YN - yPred
    braid_model.R = np.cov(yres, rowvar=False)
    # again, work in normalized space
    xres = xPred[1:] - (xPred[:-1] @ lssm.A.T + UN[:-1] @ lssm.B.T)
    braid_model.Q = np.cov(xres, rowvar=False)

    # .sId is the `MainModel`
    style_plots()
    braid_model.sId.plot_logs(saveFile=outfiles.logs_plot)
    braid_model.sId.plot_comp_graph(savepath=base_params["comp_graph_plot_basename"])

    fig, axs = plot_all(expt_data, outfiles.pred_plot)
    # plot predictions on top of previous plots
    # predict true factors X from BRAID's inferred X
    assert len(X) == len(xPred)
    # linear regression
    braid_model.linreg_braid2fsnn = LinearRegression().fit(xPred, X)
    xPred_braid2fsnn = braid_model.linreg_braid2fsnn.predict(xPred)
    t_ms = expt_data.t_ms[: len(xPred)]
    axs[0].plot(t_ms, xPred_braid2fsnn, linestyle="--")
    axs[1].plot(t_ms, zPred, linestyle="--")
    axs[2].plot(t_ms, yPred * yStd + yMean, linestyle="--")
    fig.savefig(outfiles.pred_plot)

    braid_model.sId.print_summary()
    braid_model.saveToFile(outfiles.braid_model)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    model_fit(
        snakemake.wildcards,
        snakemake.params,
        snakemake.input[0],
        snakemake.output,
        snakemake.config,
    )
