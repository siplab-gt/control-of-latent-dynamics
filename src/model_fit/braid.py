import logging
import os
import random
import types

import numpy as np

from src.model_fit import LDS

log = logging.getLogger(__name__)


def fit_lds_to_data(
    YN, ZN, UN, X_fsnn, n_x, cond_params, base_params, outfiles, config
) -> tuple[LDS, np.ndarray, np.ndarray, np.ndarray]:
    use_cpu = config["model_fit_device"].lower() == "cpu"
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        log.info("Using CPU for model fitting")
    else:
        log.info("Using GPU for model fitting")
    import BRAID

    BRAID.tools.tf_tools.setupTensorflow(cpu=use_cpu)
    import tensorflow as tf
    from BRAID.BRAIDModel import BRAIDModel, runPredict
    from BRAID.MainModel import MainModel, MainModelPrepareArgs

    extra_seed_int = sum(ord(c) for c in "model_fit")
    seed = int(cond_params["seed"]) + extra_seed_int
    log.info(f"Seeding with {cond_params.seed} + {extra_seed_int} = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    braid_model: BRAIDModel = BRAIDModel()
    # makes Cz nonlinear: 1 hidden layer with 64 units
    Cz_code = "" if base_params.linear_Cz else "Cz1HL64U"
    methodCode = (
        f"BRAID_{Cz_code}"  # Only Cz might be nonlinear, 1 Hidden layer with 64 Units.
    )
    args = MainModelPrepareArgs(methodCode)
    # args["has_Dyz"] = False
    assert args["has_Dyz"] is False
    args["epochs"] = base_params.epochs
    # TODO: BRAID warns about flat dimensions
    args["remove_flat_dims"] = False

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
        noFT=True,  # don't model feedthrough (Y->Z direct mapping)
    )

    assert braid_model.n1 == braid_model.nx == n_x
    # not modeling D (direct Y->Z mapping)
    assert braid_model.sId.model1.nft == 0
    assert braid_model.sId.ny == YN.shape[1], f"{braid_model.sId.ny=} {YN.shape[1]=}"

    # estimate Q, R
    Zhat, Yhat, Xhat, _, _, _ = runPredict(
        braid_model, YN, ZN, UN, YType="cont", ZType="cont"
    )
    # runPredict function outputs are lists of length= #steps_ahead desired for inference which is taken from the field: idSysF.step_ahead
    # Here we are only looking at 1-step-ahead predictions i.e., the lists only have one element:
    Zhat, Yhat, Xhat = Zhat[0], Yhat[0], Xhat[0]
    # Yhat has wrong shape for some reason?
    # print(f"{Yhat.shape=}")
    lssm = braid_model.getLSSM()
    # Yhat = Xhat @ lssm.C.T
    log.debug(f"{Zhat.shape=}, {Yhat.shape=}, {Xhat.shape=}")
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

    # C instead of Cfw?
    # store this to use later
    # lds.x2z_model = braid_model.sId.model1.rnn.cell.Cfw
    lds._x2z_reconstruction_info = (
        braid_model.sId.model1.rnn.cell.C.get_recreation_info()
    )
    # must get recreation info before calling, since that ruins reconstruction. Don't know why.
    Zhat_old = braid_model.sId.model1.rnn.cell.C.predict(Xhat.T).T
    assert np.allclose(Zhat_old, Zhat, rtol=1e-3), f"{Zhat_old[-10:]=}\n{Zhat[-10:]=}"
    # before BRAID's saveToFile, since it replaces Models with their weights
    braid_model.saveToFile(base_params.braid_basename + "_braid.pkl")

    if base_params.linear_Cz:
        lds.Cz = lssm.Cz

        def x2z(self, X):
            return X @ self.Cz.T
    else:

        def x2z(self, X):
            # hacky, but this is how BRAID does reconstruction internally
            x2z_model = BRAID.RegressionModel.RegressionModel(
                **self._x2z_reconstruction_info.constructor_kwargs
            ).reconstruct(self._x2z_reconstruction_info)
            return x2z_model.predict(X.T).T

    lds.x2z = types.MethodType(x2z, lds)

    assert np.allclose(Zhat_old, lds.x2z(Xhat), rtol=1e-3), (
        f"{Zhat_old[:10]=}\n{lds.x2z(Xhat)[:10]=}"
    )
    assert np.allclose(Zhat, lds.x2z(Xhat), rtol=1e-3), (
        f"{Zhat[:10]=}\n{lds.x2z(Xhat)[:10]=}"
    )

    return lds, Zhat, Yhat, Xhat
