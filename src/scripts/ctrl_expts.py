import importlib
import random
from dataclasses import dataclass

import brian2.only as b2
import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LinearRegression
from snakemake.script import snakemake

from src.ctrl import Controller
from src.helpers import seed_everything, str2bool
from src.log_redirector import redirect_output_to_file
from src.model_fit import LDS
from src.scripts.cleo_prep import CleoPrep
from src.scripts.sys_id_expts import ExptData


@dataclass
class CtrlData(ExptData):
    Xref_lds: Float[np.ndarray, "T nx"] = None
    Xhat_lds: Float[np.ndarray, "T nx"] = None
    Zhat: Float[np.ndarray, "T n_behavior"] = None
    Zref: Float[np.ndarray, "T n_behavior"] = None
    linreg_lds2fsnn: LinearRegression = None
    """to go from LDS X (stored here) to fSNN X (ground truth)"""
    C: Float[np.ndarray, "m n"] = None
    y_std: Float[np.ndarray, "m"] = None
    y_mean: Float[np.ndarray, "m"] = None

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def Xref_fsnn(self) -> Float[np.ndarray, "T n"]:
        return self.linreg_lds2fsnn.predict(self.Xref_lds)

    @property
    def Xhat_fsnn(self) -> Float[np.ndarray, "T n"]:
        return self.linreg_lds2fsnn.predict(self.Xhat_lds)

    @property
    def Yhat(self) -> Float[np.ndarray, "T m"]:
        return self.Xhat_lds @ self.C.T * self.y_std + self.y_mean

    @property
    def Yref(self) -> Float[np.ndarray, "T m"]:
        return self.Xref_lds @ self.C.T * self.y_std + self.y_mean

    @property
    def X_fsnn(self) -> Float[np.ndarray, "T n"]:
        return self.X


def ctrl_expts(
    cond_params, base_params, cleo_prep_file, lds_model_file, outfile, config
):
    # import here so not needed to use CtrlData class
    import tensorflow as tf
    from BRAID.BRAIDModel import BRAIDModel

    print(f"{base_params=}")
    print(base_params.keys())

    # careful to load before setting seed, since loading sets seed as side effect
    cleo_prep = CleoPrep.load(cleo_prep_file)
    cleo_prep.reset()

    seed = seed_everything(int(cond_params["seed"]), "ctrl_expts")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)

    k_opto = int(cond_params["k"])
    u_max = float(base_params["u_max_mWmm2"])
    t_intertrial = float(base_params.expts["t_intertrial_ms"]) * b2.ms
    t_trial = float(base_params.expts["t_trial_ms"]) * b2.ms
    n_trials = int(base_params["n_trials_ctrl"])
    report_period = config["report_period_s"] * b2.second

    lds = LDS.load(lds_model_file)

    if str2bool(cond_params.model_task_inputs):
        k_task = cleo_prep.fsnn.k_task
    else:
        k_task = 0
    assert k_opto + k_task == lds.k, f"{lds.k=} {k_opto=} {k_task=}"

    # separate files so one can change without causing the other to be rerun
    ctrl_type = cond_params["ctrl_type"]
    ControllerType = getattr(
        importlib.import_module(f"src.ctrl.{ctrl_type}"),
        f"{ctrl_type.upper()}Controller",
    )
    ctrl_proc: Controller = ControllerType(
        rng=rng,
        sim=cleo_prep.sim,
        u_max_mWmm2=u_max,
        lds=lds,
        add_latency=str2bool(cond_params.latency),
        baseline_latency=float(base_params["baseline_latency_ms"]) * b2.ms,
        sample_period=float(base_params["sample_period_ms"]) * b2.ms,
        u_task=cleo_prep.fsnn.u_task_,
        # horizon will be overwritten in OL to cover whole trial
        horizon=int(base_params["ctrl_horizon_ms"])
        // int(base_params["sample_period_ms"]),
        Qcost=float(base_params.Q_cost),
        Rcost=float(base_params.R_cost),
        reference=lambda t: np.zeros((*np.shape(t), lds.n)),
        cleo_prep=cleo_prep,
        state_estimation="ground truth"
        if cond_params.setting == "sufficient_stim"
        else "kalman",
    )

    if ctrl_proc.add_latency:
        ctrl_proc.compute_latency = ctrl_proc.measure_latency()
        print(f"measured control computation latency: {ctrl_proc.compute_latency}")

    cleo_prep.sim.set_io_processor(ctrl_proc)

    i_trial_boundaries = []
    for i_trial in range(n_trials):
        print(f"running control trial {i_trial}")
        t_now = cleo_prep.sim.network.t
        ctrl_proc.update_with_trial_info(
            t_now, t_now + t_intertrial, t_now + t_intertrial + t_trial
        )
        cleo_prep.run(t_intertrial, report_period=report_period)
        i_start = cleo_prep.fsnn.state_mon.X.shape[1]
        condition = cleo_prep.fsnn.sample_condition(rng)
        cleo_prep.fsnn.set_condition(condition)
        cleo_prep.run(t_trial, report_period=report_period)
        i_end = cleo_prep.fsnn.state_mon.X.shape[1]
        i_trial_boundaries.append((i_start, i_end))

    ctrl_data = CtrlData.from_cleo_prep(
        cleo_prep,
        np.array(ctrl_proc.spike_counts),
        cleo_prep.sim.io_processor.t_samp,
        i_trial_boundaries,
    )
    print("computing Zref and Zhat from Xref and Xhat")
    Zref = lds.x2z(ctrl_proc.Xref)
    Zhat = lds.x2z(ctrl_proc.Xhat)
    ctrl_data.set(
        C=lds.C,
        Xref_lds=ctrl_proc.Xref,
        Xhat_lds=ctrl_proc.Xhat,
        linreg_lds2fsnn=lds.linreg_lds2fsnn,
        y_std=lds.y_std,
        y_mean=lds.y_mean,
        Zhat=Zhat,
        Zref=Zref,
        compute_latency_ms=ctrl_proc.compute_latency / b2.ms
        if ctrl_proc.add_latency
        else 0,
    )
    ctrl_data.save(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    with redirect_output_to_file(snakemake.log[0]):
        ctrl_expts(
            snakemake.wildcards,
            snakemake.params,
            snakemake.input.cleo_prep,
            snakemake.input.lds_model,
            snakemake.output[0],  # outfile
            snakemake.config,  # for non-essential params
        )
