import importlib
import logging
import random
from dataclasses import dataclass

import brian2.only as b2
import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LinearRegression
from snakemake.script import snakemake

from src.ctrl import Controller
from src.ctrl.solvers import SOLVERS
from src.helpers import (
    preprocess_params,
    seed_everything,
    setup_debugger,
    setup_logging,
    str2bool,
)
from src.model_fit import LDS
from src.scripts.cleo_prep import CleoPrep
from src.scripts.sys_id_expts import ExptData

log = logging.getLogger(__name__)


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

    compute_latency_mean_ms: float = None
    compute_latency_std_ms: float = None
    total_latency_ms: float = None

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def Xref_fsnn(self) -> Float[np.ndarray, "T n"]:
        if self.Xref_lds is None or np.size(self.Xref_lds) == 0:
            return None
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
    cond_params,
    base_params,
    cleo_prep_file,
    lds_model_file,
    out_data_file,
    config,
):
    # import here so not needed to use CtrlData class
    import tensorflow as tf
    from BRAID.BRAIDModel import BRAIDModel

    log.debug(f"{base_params=}")

    # careful to load before setting seed, since loading sets seed as side effect
    cleo_prep = CleoPrep.load(cleo_prep_file)

    seed = seed_everything(int(cond_params["seed"]), "ctrl_expts")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)

    k_opto = cond_params.k
    u_max = base_params.u_max_mWmm2
    n_trials = base_params.n_trials_ctrl

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
    add_latency = (cond_params.ctrl_type != "OL") and str2bool(cond_params.latency)
    ctrl_proc: Controller = ControllerType(
        sim=cleo_prep.sim,
        u_max_mWmm2=u_max,
        lds=lds,
        ref_type=base_params.ref_type,
        add_latency=add_latency,
        baseline_latency=base_params.baseline_latency,
        # use separate RNG for latency measurement/sampling so trials are same regardless
        latency_rng=np.random.default_rng(seed + 18010630),
        sample_period=cleo_prep.fsnn.dt * base_params.subsample_factor,
        # in OL, ctrl_horizon is pre-trial control time; horizon includes whole trial
        ctrl_horizon=base_params.ctrl_horizon,
        Qcost=float(base_params.Q_cost),
        Rcost=float(base_params.R_cost),
        cleo_prep=cleo_prep,
        state_estimation="ground truth"
        if cond_params.setting == "sufficient_stim"
        else "kalman",
        signals=cond_params.signals,
    )

    # Test latency for each solver and order them from fastest to slowest
    log.info("Testing latency for all solvers...")
    solver_latencies = []
    for solver in SOLVERS:
        ctrl_proc.solvers = [solver]
        mean_lat, std_lat = ctrl_proc.measure_latency()
        solver_latencies.append((solver, mean_lat, std_lat))
        log.info(
            f"  Solver {solver.name}: {mean_lat / b2.ms:.2f} +/- {std_lat / b2.ms:.2f} ms"
        )

    # Order solvers from fastest to slowest
    solver_latencies.sort(key=lambda x: x[1])  # Sort by mean latency
    ordered_solvers = [s[0] for s in solver_latencies]
    fastest_mean_lat = solver_latencies[0][1]
    fastest_std_lat = solver_latencies[0][2]

    log.info(f"Fastest solver: {ordered_solvers[0].name}")
    log.info(f"Solver order: {[s.name for s in ordered_solvers]}")

    # Update controller with ordered solvers and fastest solver's latency stats
    ctrl_proc.solvers = ordered_solvers

    if add_latency:
        ctrl_proc.compute_latency_mean = fastest_mean_lat
        ctrl_proc.compute_latency_std = fastest_std_lat
    else:
        ctrl_proc.compute_latency_mean = 0 * b2.ms
        ctrl_proc.compute_latency_std = 0 * b2.ms

    cleo_prep.sim.set_io_processor(ctrl_proc)

    log.info(f"running {n_trials} control trials")
    cleo_prep.fsnn.run_trials(
        n_trials,
        mode="test",
        rng=rng,
        record=True,
        n_init_trials=0,
        n_settle_trials=0,
        ctrl=True,
    )

    spike_counts = np.array(ctrl_proc.spike_counts) if ctrl_proc.spike_counts else None

    ctrl_data = CtrlData.from_cleo_prep(
        cleo_prep,
        spike_counts,
        cleo_prep.sim.io_processor.t_samp,
        base_params.subsample_factor,
    )
    log.info("extracting Zref and Zhat")
    Zref = ctrl_proc.Zref * lds.z_std + lds.z_mean
    Zhat = ctrl_proc.Zhat * lds.z_std + lds.z_mean
    ctrl_data.set(
        C=lds.C,
        Xref_lds=ctrl_proc.Xref,
        Xhat_lds=ctrl_proc.Xhat,
        linreg_lds2fsnn=lds.linreg_lds2fsnn,
        y_std=lds.y_std,
        y_mean=lds.y_mean,
        Zhat=Zhat,
        Zref=Zref,
        compute_latency_mean_ms=ctrl_proc.compute_latency_mean / b2.ms,
        compute_latency_std_ms=ctrl_proc.compute_latency_std / b2.ms,
        total_latency_ms=ctrl_proc.latency_mean / b2.ms,
    )
    ctrl_data.save(out_data_file)


if __name__ == "__main__" and "snakemake" in locals():
    fh = setup_logging(snakemake.config, snakemake.log[0])
    setup_debugger(snakemake.config["debug"])

    ctrl_expts(
        preprocess_params(snakemake.wildcards),
        preprocess_params(snakemake.params),
        snakemake.input.cleo_prep,
        snakemake.input.lds_model,
        snakemake.output.data,  # outfile
        preprocess_params(snakemake.config),  # for non-essential params
    )
