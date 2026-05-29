import logging
import os
import random
import re
import types

import numpy as np

from src.model_fit import LDS
from src.model_fit.torch_cz import TorchNonlinearCz, torch_cz_predict

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
    # makes Cz nonlinear. Default 1 hidden layer with 16 units (down from 64
    # after Sweep 6 found that Cz1HL64U overfits to the low-amplitude
    # sys_id training distribution, then collapses under closed-loop drive
    # which is 5-6× higher amplitude — Sweep 7 diagnosis, scratch, not committed).
    # Configurable via `cz_hidden_units` in the per-setting model_fit overlay.
    if base_params.linear_Cz:
        Cz_code = ""
    else:
        cz_hidden_units = getattr(base_params, 'cz_hidden_units', 16)
        Cz_code = f"Cz1HL{int(cz_hidden_units)}U"
    methodCode = f"BRAID_{Cz_code}_{base_params.steps_ahead_model_code}"
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

    # Post-fit stability projection (nonlinear-Cz only). Sweep 6 forensic
    # diagnosis (scratch, not committed) found nonlinear joint
    # training pushes A's spectral radius to ~1.000 on some seeds — the
    # controller then over-drives a near-unit-modulus mode through the
    # nonlinear Cz, blowing up closed-loop Z prediction. The legacy
    # safety net in ctrl_expts.py (line 128) only triggers at |λ|≥1.0;
    # tighten here to |λ|≤0.97 for nonlinear-Cz cells.
    if not base_params.linear_Cz:
        max_eig = float(np.max(np.abs(np.linalg.eigvals(lds.A))))
        if max_eig > 0.97:
            scale = 0.97 / max_eig
            lds.A = lds.A * scale
            lds.B = lds.B * scale
            log.warning(
                f"Nonlinear-Cz BRAID: spectral radius {max_eig:.4f} > 0.97; "
                f"scaling A and B by {scale:.4f} to enforce |λ|≤0.97."
            )

    # .sId is the `MainModel`
    outdir = os.path.dirname(outfiles.lds_model)
    braid_basename = outdir + "/model_fit_braid"
    braid_model.sId.plot_logs(saveFile=braid_basename + "_logs_plot.png")
    comp_graph_plot_basename = braid_basename + "_comp_graph"
    braid_model.sId.plot_comp_graph(savepath=comp_graph_plot_basename)
    braid_model.sId.print_summary()

    # C instead of Cfw?
    # store this to use later
    # lds.x2z_model = braid_model.sId.model1.rnn.cell.Cfw
    lds._x2z_reconstruction_info = (
        braid_model.sId.model1.rnn.cell.C.get_recreation_info()
    )
    # Port the trained nonlinear Cz to a torch nn.Module here — BEFORE
    # `braid_model.saveToFile` runs (which mutates `braid_model.sId.model1`
    # in-place from a live RNN to a ReconstructionInfo container and drops
    # the `.rnn` attribute, breaking weight access). The port is a pure
    # weight copy; the resulting `cz_torch_module` is independent of the
    # subsequent BRAID save and is dilled along with the LDS.
    if not base_params.linear_Cz:
        m_cz = re.match(r"Cz(\d+)HL(\d+)U", Cz_code)
        if not m_cz:
            raise ValueError(
                f"Cannot parse Cz topology from Cz_code={Cz_code!r}; "
                f"expected 'Cz<N>HL<U>U' pattern when linear_Cz=False."
            )
        n_hidden = int(m_cz.group(1))
        hidden_units = int(m_cz.group(2))
        lds.cz_torch_module = TorchNonlinearCz.from_braid_regression(
            braid_model.sId.model1.rnn.cell.C,
            n_in=braid_model.nx,
            n_out=ZN.shape[1],
            hidden_units=hidden_units,
            num_hidden_layers=n_hidden,
            activation='relu',
        )

    # must get recreation info before calling, since that ruins reconstruction. Don't know why.
    Zhat_old = braid_model.sId.model1.rnn.cell.C.predict(Xhat.T).T
    tol_params = dict(rtol=1e-2, atol=1e-5)
    assert np.allclose(Zhat_old, Zhat, **tol_params), (
        f"{Zhat_old[-10:]=}\n{Zhat[-10:]=}\n{Zhat[~np.isclose(Zhat_old, Zhat, **tol_params)]=}"
    )
    # before BRAID's saveToFile, since it replaces Models with their weights
    braid_model.saveToFile(braid_basename + "_braid.pkl")

    if base_params.linear_Cz:
        lds.Cz = lssm.Cz

        def x2z(self, X):
            return X @ self.Cz.T
    else:
        # Sanity-check the torch port matches BRAID's predictions on the
        # training trajectory. Threshold reflects observed fp32 round-trip
        # noise between TF and torch matmul (~1e-5 to 1e-4 in practice).
        Y_torch = torch_cz_predict(lds.cz_torch_module, Xhat.T)  # (m_z, T)
        port_diff = np.abs(Y_torch.T - Zhat).max()
        log.info(
            f"Nonlinear Cz: torch port max diff vs BRAID on training "
            f"Xhat = {port_diff:.2e}"
        )
        if port_diff > 1e-3:
            raise RuntimeError(
                f"Torch nonlinear Cz port diverges from BRAID by "
                f"{port_diff:.2e} (threshold 1e-3); weight transfer is broken."
            )

        def x2z(self, X):
            # Numpy entry point preserved for legacy callers (analysis,
            # base Controller introspection). Controllers should prefer
            # `lds.cz_torch_module` directly for differentiable rollouts.
            x2z_model = BRAID.RegressionModel.RegressionModel(
                **self._x2z_reconstruction_info.constructor_kwargs
            ).reconstruct(self._x2z_reconstruction_info)
            return x2z_model.predict(X.T).T

    lds.x2z = types.MethodType(x2z, lds)

    assert np.allclose(Zhat_old, lds.x2z(Xhat), **tol_params), (
        f"{Zhat_old[-10:]=}\n{Zhat[-10:]=}\n{Zhat[~np.isclose(Zhat_old, Zhat, **tol_params)]=}"
    )

    return lds, Zhat, Yhat, Xhat
