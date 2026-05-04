import importlib
import logging
import random
from dataclasses import dataclass

import brian2.only as b2
import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LinearRegression

from src.ctrl import Controller
from src.ctrl.solvers import get_solvers
from src.helpers import (
    PassthroughIOProcessor,  # Required for dill to deserialize CleoPrep with LFP
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
    compute_latency_nonsolve_ms: float = None
    total_latency_ms: float = None

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def _predict_fsnn_tolerating_nan(
        model, X: np.ndarray
    ) -> np.ndarray:
        """Run linreg prediction; preserve NaN rows instead of crashing.

        Under pathological plant configs (e.g., fSNN destabilization at
        high-k under v3 plant, see D6 in v3_opsin_deviations.md), the USID
        closed-loop observer can diverge and leave NaN in Xhat_lds / Xref_lds.
        sklearn's LinearRegression.predict() refuses to ingest NaN and
        crashes analyze_ctrl_data. We do the predict only on the finite rows
        and slot NaN back in for the rest, so downstream R² computation can
        proceed and surface the pathology as NaN metrics rather than a
        pipeline crash.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        out = np.full((X.shape[0], model.coef_.shape[0]), np.nan)
        finite_rows = np.isfinite(X).all(axis=1)
        if finite_rows.any():
            out[finite_rows] = model.predict(X[finite_rows])
        return out

    @property
    def Xref_fsnn(self) -> Float[np.ndarray, "T n"]:
        if self.Xref_lds is None or np.size(self.Xref_lds) == 0:
            return None
        return self._predict_fsnn_tolerating_nan(self.linreg_lds2fsnn, self.Xref_lds)

    @property
    def Xhat_fsnn(self) -> Float[np.ndarray, "T n"]:
        return self._predict_fsnn_tolerating_nan(self.linreg_lds2fsnn, self.Xhat_lds)

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

    # Check stability and scale down A and B if unstable
    eigenvalues = np.linalg.eigvals(lds.A)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    if max_eigenvalue >= 1.0:
        scale_factor = 0.99 / max_eigenvalue
        lds.A *= scale_factor
        lds.B *= scale_factor
        log.warning(
            f"LDS unstable (max eigenvalue: {max_eigenvalue:.4f}). Scaled A and B by {scale_factor:.4f}"
        )
        if hasattr(lds, 'K1'):
            lds.A11 *= scale_factor
            lds.A22 *= scale_factor
            lds.K1 *= scale_factor
            lds.K2 *= scale_factor
            lds.B1_prime *= scale_factor
            lds.B2_prime *= scale_factor
            if lds.A12 is not None:
                lds.A12 *= scale_factor
            log.info(f"Scaled observer subspace matrices by {scale_factor:.4f}")

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
    add_latency = (cond_params.ctrl_type not in ["NULL", "OL"]) and str2bool(
        cond_params.latency
    )
    base_lat = base_params.baseline_latency
    ctrl_kwargs = dict(
        sim=cleo_prep.sim,
        u_max_mWmm2=u_max,
        lds=lds,
        ref_type=base_params.ref_type,
        add_latency=add_latency,
        baseline_latency=base_lat,
        latency_rng=np.random.default_rng(seed + 18010630),
        sample_period=cleo_prep.fsnn.dt * base_params.subsample_factor,
        ctrl_horizon=cond_params.ctrl_horizon,
        Qcost=float(base_params.Q_cost),
        Rcost=float(base_params.R_cost),
        cleo_prep=cleo_prep,
        state_estimation=(
            "ground truth" if cond_params.setting == "sufficient_stim"
            else "observer" if hasattr(lds, "K1")
            else "kalman"
        ),
        signals=cond_params.signals,
    )
    if ctrl_type == "NLMPC":
        ctrl_kwargs['nlmpc_iters'] = int(cond_params.nlmpc_iters)
        ctrl_kwargs['nlmpc_lr'] = float(base_params.nlmpc_lr)
    ctrl_proc: Controller = ControllerType(**ctrl_kwargs)

    if ctrl_type not in ("NULL", "NLMPC"):
        # Test latency for each solver and order them from fastest to slowest
        log.info("Testing all solvers...")
        solver_latencies = []
        for solver in get_solvers(base_params.solvers, base_params.solver_args):
            ctrl_proc.solvers = [solver]
            # use large time limit for testing
            solver.generic_args["time_limit"] = 2
            try:
                mean_lat, std_lat, nonsolve_lat = ctrl_proc.measure_latency()
            except Exception as e:
                log.warning(f"  Solver {solver.name} failed latency test: {e}")
                # very large latency
                mean_lat, std_lat, nonsolve_lat = b2.ms * 1e9, b2.ms * 1e9, 0 * b2.ms
                # just in case this doesn't get reset
                ctrl_proc._measuring_latency = False
            solver_latencies.append((solver, mean_lat, std_lat, nonsolve_lat))
            log.info(
                f"  Solver {solver.name}: {mean_lat / b2.ms:.2f} +/- {std_lat / b2.ms:.2f} ms"
            )
            solve_limit = base_params.time_limit - base_lat - nonsolve_lat
            if solve_limit <= 0 * b2.ms:
                log.warning(
                    f"  Solver {solver.name} has non-solve latency {nonsolve_lat / b2.ms:.2f} ms "
                    f"which exceeds time budget of {base_params.time_limit - base_lat}."
                )
                solve_limit = 1 * b2.ms  # minimal time limit
            solver.generic_args["time_limit"] = solve_limit / b2.second

        # Order solvers from fastest to slowest
        solver_latencies.sort(key=lambda x: x[1])  # Sort by mean latency
        ordered_solvers = [s[0] for s in solver_latencies]
        fastest_mean_lat = solver_latencies[0][1]
        fastest_std_lat = solver_latencies[0][2]
        fastest_nonsolve_lat = solver_latencies[0][3]

        log.info(f"Fastest solver: {ordered_solvers[0]}")
        log.info(f"Solver order: {[s.name for s in ordered_solvers]}")

        # Update controller with ordered solvers and fastest solver's latency stats
        ctrl_proc.solvers = ordered_solvers

    if ctrl_type == "NLMPC":
        # NLMPC uses its own latency measurement (no CVXPY solvers)
        if add_latency:
            log.info("Measuring NLMPC latency...")
            fastest_mean_lat, fastest_std_lat, fastest_nonsolve_lat = ctrl_proc.measure_latency()
            log.info(f"NLMPC latency: {fastest_mean_lat / b2.ms:.2f} +/- {fastest_std_lat / b2.ms:.2f} ms")
            ctrl_proc.compute_latency_mean = fastest_mean_lat
            ctrl_proc.compute_latency_std = fastest_std_lat
            ctrl_proc.compute_latency_nonsolve = fastest_nonsolve_lat
        else:
            ctrl_proc.compute_latency_mean = 0 * b2.ms
            ctrl_proc.compute_latency_std = 0 * b2.ms
            ctrl_proc.compute_latency_nonsolve = 0 * b2.ms
    elif add_latency:
        if ctrl_type == "LQMPC":
            expected_solve_time = fastest_mean_lat - fastest_nonsolve_lat
            if (
                expected_solve_time
                > ordered_solvers[0].generic_args["time_limit"] * b2.second
            ):
                log.warning(
                    f"The expected solve time of {expected_solve_time} exceeds the time limit of "
                    f"{ordered_solvers[0].generic_args['time_limit'] * 1000:.2f} ms. "
                )
        ctrl_proc.compute_latency_mean = fastest_mean_lat
        ctrl_proc.compute_latency_std = fastest_std_lat
        ctrl_proc.compute_latency_nonsolve = fastest_nonsolve_lat
    else:
        if ctrl_type != "NULL":
            for solver in ctrl_proc.solvers:
                solver.generic_args["time_limit"] = 2  # large time limit
        ctrl_proc.compute_latency_mean = 0 * b2.ms
        ctrl_proc.compute_latency_std = 0 * b2.ms
        ctrl_proc.compute_latency_nonsolve = 0 * b2.ms

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
        cleo_prep_file,
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
        compute_latency_nonsolve_ms=ctrl_proc.compute_latency_nonsolve / b2.ms,
        total_latency_ms=ctrl_proc.latency_mean / b2.ms,
    )
    ctrl_data.save(out_data_file)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

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
