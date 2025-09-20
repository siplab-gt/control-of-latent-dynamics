import os
import random
from dataclasses import dataclass

import BRAID
import dill
import numpy as np
from matplotlib import pyplot as plt

from src.helpers import str2bool
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
    from BRAID.BRAIDModel import BRAIDModel
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
    Y = expt_data.Y[:T]
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
    # YtrainN, YtestN = (Ytrain - yMean) / yStd, (Ytest - yMean) / yStd
    YN = (Y - yMean) / yStd
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
        n1=2,
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

    # .sId is the `MainModel`
    braid_model.sId.plot_logs(saveFile=outfiles.logs_plot)
    braid_model.sId.plot_comp_graph(savepath=base_params["comp_graph_plot_basename"])

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
