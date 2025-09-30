import os
import random
import types

import numpy as np

from src.model_fit import LDS


def fit_lds_to_data(
    YN, ZN, UN, X_fsnn, n_x, config, cond_params, base_params, outfiles
) -> tuple[LDS, np.ndarray, np.ndarray, np.ndarray]:
    use_cpu = config["model_fit_device"].lower() == "cpu"
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Using CPU for model fitting")
    else:
        print("Using GPU for model fitting")
    import BRAID

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

    braid_model: BRAIDModel = BRAIDModel()
    methodCode = (
        "BRAID_Cz1HL64U"  # Only Cz will be nonlinear, 1 Hidden layer with 64 Units.
    )
    args = MainModelPrepareArgs(methodCode)
    args["epochs"] = int(base_params.model_fit["epochs"])

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

    assert braid_model.n1 == braid_model.nx == n_x

    # estimate Q, R
    Zhat, Yhat, Xhat, _, _, _ = runPredict(
        braid_model, YN, ZN, UN, YType="cont", ZType="cont"
    )
    # runPredict function outputs are lists of length= #steps_ahead desired for inference which is taken from the field: idSysF.step_ahead
    # Here we are only looking at 1-step-ahead predictions i.e., the lists only have one element:
    Zhat, Yhat, Xhat = Zhat[0], Yhat[0], Xhat[0]
    print(f"{Zhat.shape=}, {Yhat.shape=}, {Xhat.shape=}")
    lssm = braid_model.getLSSM()
    yres = YN - Yhat
    R = np.cov(yres, rowvar=False)
    # again, work in normalized space
    xres = Xhat[1:] - (Xhat[:-1] @ lssm.A.T + UN[:-1] @ lssm.B.T)
    Q = np.cov(xres, rowvar=False)
    lds = LDS(
        A=lssm.A,
        B=lssm.B,
        C=lssm.C,
        Q=Q,
        R=R,
        k_opto=int(cond_params["k"]),
    )

    # .sId is the `MainModel`
    braid_model.sId.plot_logs(saveFile=base_params.braid_basename + "_logs_plot.png")
    comp_graph_plot_basename = base_params.braid_basename + "_comp_graph"
    braid_model.sId.plot_comp_graph(savepath=comp_graph_plot_basename)
    braid_model.sId.print_summary()
    braid_model.saveToFile(base_params.braid_basename + "_braid.pkl")

    # store this to use later
    lds.x2z_model = braid_model.sId.model1.rnn.cell.Cfw

    def x2z(self, X):
        return self.x2z_model.predict(X.T).T

    lds.x2z = types.MethodType(x2z, lds)

    return lds, Zhat, Yhat, Xhat
