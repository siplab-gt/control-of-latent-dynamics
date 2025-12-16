import logging
import time
from abc import abstractmethod
from math import ceil, floor

import brian2.only as b2
import cleo
import cvxpy as cp
import numpy as np
from attrs import define, field
from brian2 import Quantity, mm2, ms, mwatt, second
from jaxtyping import Float
from scipy import stats

from src.ctrl.solvers import Solver
from src.helpers import array_vals_by_t
from src.model_fit import LDS
from src.scripts.cleo_prep import CleoPrep

log = logging.getLogger(__name__)


@define
class Controller(cleo.ioproc.LatencyIOProcessor):
    sim: cleo.CLSimulator = None
    u_max_mWmm2: float = None
    add_latency: bool = True
    baseline_latency: Quantity = None
    compute_latency_mean: Quantity = None
    compute_latency_std: Quantity = None
    latency_rng: np.random.Generator = None
    lds: LDS = None
    ref_type: str = None
    """callable with signature t -> task inputs (shape ([T], k_task) array)"""
    ctrl_horizon: Quantity = None
    """control horizon (with time unit). In OL case, this is pre-trial control time."""
    Qcost: float = None
    Rcost: float = None
    cleo_prep: CleoPrep = None
    state_estimation: str = None
    signals: str = None
    """spikes, LFP, or both"""
    solvers: list[Solver] = None
    """list of solvers ordered from fastest to slowest"""

    spike_counts: list | None = None
    _Xhat: list = field(factory=list)
    _R: list = field(factory=list)
    xpred: np.ndarray = None
    Ppred: np.ndarray = None

    Uopt_opto: cp.Variable = None
    """optimized u trajectory over horizon"""
    Uopt_task: cp.Parameter = None
    """task inputs over horizon"""
    Xopt: cp.Variable = None
    """x trajectory over horizon including initial state"""
    Ropt: cp.Parameter = None
    """reference trajectory over horizon"""
    x0opt: cp.Parameter = None
    """initial state parameter for optimization"""
    prob: cp.Problem = None
    """cvxpy optimization problem"""

    i_current_trial: int = 0
    _measuring_latency: bool = False
    _cached_ref_trajectory: np.ndarray = None
    """to avoid having to zscore repeatedly (which made memory explode).
    should be once for all trials, since all trials are run as a block."""

    def __attrs_post_init__(self):
        self.reset()

    # don't know why I need to make this a field to avoid attrs error
    _needs_off_schedule_sample: bool = False

    def reset(self):
        super().reset()
        self.xpred = np.zeros(self.n)
        self.Ppred = np.eye(self.n)  # initial covariance
        self.spike_counts = [] if self.signals in ["spikes", "both"] else None
        self._Xhat = []
        self._R = []
        self._cached_ref_trajectory = None

    def init_optimization(self):
        self.Uopt_opto = cp.Variable((self.n_horizon, self.k_opto))
        if self.k_task > 0:
            self.Uopt_task = cp.Parameter((self.n_horizon, self.k_task))
        self.Xopt = cp.Variable((self.n_horizon + 1, self.n))
        self.Ropt = cp.Parameter((self.n_horizon, self.n_r))
        self.x0opt = cp.Parameter(self.n)
        Uopt_opto_unzscored = (
            cp.multiply(self.Uopt_opto, self.lds.u_std[None, : self.k_opto])
            + self.lds.u_mean[None, : self.k_opto]
        )
        ctrl_var_traj = self.Xopt if self.ref_type == "X" else self.lds.x2z(self.Xopt)
        cost = (
            cp.sum_squares(ctrl_var_traj[1:] - self.Ropt) * self.Qcost / self.n_r
            + cp.sum_squares(Uopt_opto_unzscored) * self.Rcost / self.k_opto
        )

        if self.k_task > 0:
            task_input = self.Uopt_task @ self.B_task.T
        else:
            task_input = 0
        constraints = [
            # initial state
            self.Xopt[0, :] == self.x0opt,
            # dynamics
            self.Xopt[1:, :]
            == self.Xopt[:-1, :] @ self.A.T
            + self.Uopt_opto @ self.B_opto.T
            + task_input,
            Uopt_opto_unzscored >= 0,
            Uopt_opto_unzscored <= self.u_max_mWmm2,
        ]
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

        self.Uopt_opto.value = self.zscore(np.zeros((self.n_horizon, self.k)), "u")[
            :, : self.k_opto
        ]
        self.Xopt.value = np.zeros((self.n_horizon + 1, self.n))

        assert self.prob.is_dcp(dpp=False), "problem not DCP"
        assert self.prob.is_dcp(dpp=True), "problem not DPP"

    def update_opt_problem(self, xhat, t_samp):
        self.x0opt.value = xhat[:]
        if self.k_task > 0:
            self.Uopt_task.value = self.u_task_zscored(
                t_samp + np.arange(self.n_horizon) * self.sample_period
            )
        t_horizon = self.t_now + np.arange(1, self.n_horizon + 1) * self.sample_period
        self.Ropt.value = self.reference_zscored(t_horizon)

    @property
    def Xhat(self) -> Float[np.ndarray, "T n"]:
        """in LDS (z-scored) space"""
        return np.array(self._Xhat)

    @property
    def Xref(self) -> Float[np.ndarray, "T n"]:
        """in LDS (z-scored) space"""
        if self.ref_type == "Z":
            return None
        assert self.ref_type == "X"
        return np.array(self._R)

    @property
    def Zref(self) -> Float[np.ndarray, "T m_z"]:
        """in LDS (z-scored) space"""
        if self.ref_type == "X":
            return self.lds.x2z(self.Xref)
        assert self.ref_type == "Z"
        return np.array(self._R)

    @property
    def Zhat(self) -> Float[np.ndarray, "T m_z"]:
        """in LDS (z-scored) space"""
        return self.lds.x2z(self.Xhat)

    @property
    def Yhat(self) -> Float[np.ndarray, "T m"]:
        """in original (unnormalized) space"""
        return self.unzscore(self.Xhat @ self.C.T, "y")

    @property
    def t_now(self) -> Quantity:
        """rounded to nearest sample period to avoid floating point issues"""
        return round(self.sim.network.t / self.sample_period) * self.sample_period

    @property
    def n_horizon(self) -> int:
        return int(round(self.ctrl_horizon / self.sample_period))

    @property
    def n_r(self) -> int:
        return self.lds.m_z if self.ref_type == "Z" else self.n

    @property
    def subsample_factor(self) -> int:
        ssf = self.sample_period / b2.defaultclock.dt
        assert ssf == int(ssf), "sample period not integer multiple of b2 dt"
        return int(ssf)

    def measure_latency(self) -> tuple[b2.Quantity, b2.Quantity]:
        """returns mean, std of control computation (with Brian unit)"""
        self._measuring_latency = True
        durations = []
        xpred = np.zeros(self.n)
        Ppred = np.eye(self.n)
        u = self.latency_rng.normal(self.lds.u_mean, self.lds.u_std, self.k)
        for i in range(100):
            avg_rate = 20 / second
            y = self.latency_rng.poisson(lam=avg_rate * self.sample_period)

            start = time.perf_counter()
            xhat, P = self.update_state(xpred, y, Ppred)
            uopto = self.compute_control(xhat)
            u = np.concatenate((uopto, self.u_task_zscored(10 * ms)))
            xpred, Ppred = self.predict_state(xhat, u, P)
            end = time.perf_counter()
            durations.append(end - start)

        # take off first 5 to let things warm up
        log.debug(f"latency measurement durations (s):\n{durations}")
        durations = np.array(durations[5:])
        log.debug(f"{stats.describe(durations, bias=False)=}")
        self._measuring_latency = False
        # to make robust to outliers from unreliable compute speed on cluster,
        # estimate mean and std using median and MAD
        median = np.median(durations)
        mad = np.median(np.abs(durations - median))
        return median * b2.second, mad / 0.6745 * b2.second

    def zscore(self, var: np.ndarray, varname: str):
        mean = getattr(self.lds, f"{varname}_mean")
        std = getattr(self.lds, f"{varname}_std")
        return (var - mean) / std

    def unzscore(self, var: np.ndarray, varname: str):
        mean = getattr(self.lds, f"{varname}_mean")
        std = getattr(self.lds, f"{varname}_std")
        return var * std + mean

    def update_state(self, xpred, y, Ppred) -> tuple[np.ndarray, np.ndarray]:
        """returns (xhat, P)"""
        assert np.linalg.norm(Ppred) < 1e100, "Ppred exploded"
        y = self.zscore(y, "y")
        yerr = y - self.C @ xpred  # innovation

        # use pinv instead of inv in case of singular matrix (where C has an all-0 zero)
        # K = Ppred @ self.C.T @ np.linalg.pinv(self.C @ Ppred @ self.C.T + self.R)

        # More Robust: Use solve
        S = self.C @ Ppred @ self.C.T + self.R
        P_CT = Ppred @ self.C.T
        # Solve the linear system S * X = P_CT for X
        # np.linalg.solve expects A and B such that AX = B
        # Since S is symmetric, solving S * X = P_CT is safe.
        K = np.linalg.solve(S, P_CT).T

        xhat = xpred + K @ yerr
        P = (np.eye(self.n) - K @ self.C) @ Ppred
        return xhat, P

    def predict_state(self, xhat, u, P):
        """u is assumed to be z-scored. returns (xpred, Ppred)"""
        # P_pred = sim.A @ sim.P @ sim.A.T + sim.Q
        xpred = self.A @ xhat + self.B @ u
        Ppred = self.A @ P @ self.A.T + self.Q

        return xpred, Ppred

    def solve_optimization(self, ctrl_type: str) -> None:
        """Solve the optimization problem with fallback through ordered solvers.

        Args:
            ctrl_type: Control type string (e.g., 'LQMPC', 'OL') for solver-specific settings

        Raises:
            RuntimeError: If all solvers fail
        """
        for i, solver in enumerate(self.solvers):
            try:
                solve_kwargs = solver.get_solve_kwargs(ctrl_type)
                self.prob.solve(**solve_kwargs)

                if self.prob.status not in ["infeasible", "unbounded"]:
                    if i > 0:
                        log.warning(
                            f"Solver {solver.name} succeeded after {i} failed attempts"
                        )
                    return
                else:
                    log.warning(
                        f"Solver {solver.name} returned status: {self.prob.status}\n"
                        f"solver_stats = {self.prob.solver_stats}"
                    )
            except Exception as e:
                log.warning(
                    f"Solver {solver.name} failed with error: {e}\n"
                    f"solver_stats = {self.prob.solver_stats}"
                )
                if i == len(self.solvers) - 1:
                    raise

        raise RuntimeError(f"All solvers failed. Last status: {self.prob.status}")

    @abstractmethod
    def compute_control(self, xhat):
        pass

    def reference_zscored(self, t: Quantity) -> Quantity:
        if self._measuring_latency:
            return np.zeros((*np.shape(t), self.n_r))

        # Cache reference trajectory once for all trials
        if self._cached_ref_trajectory is None:
            fsnn = self.cleo_prep.fsnn
            if self.ref_type == "Z":
                ref = self.zscore(fsnn.fout_ta.values, "z")
            elif self.ref_type == "X":
                ref = self.zscore(fsnn.ytilde_ta.values, "x")
            self._cached_ref_trajectory = ref

        return array_vals_by_t(self._cached_ref_trajectory, t, self.base_dt)

    def u_task(self, t: Quantity) -> np.ndarray:
        fsnn = self.cleo_prep.fsnn
        return array_vals_by_t(fsnn.fin_ta.values, t, self.base_dt)

    @property
    def t_ctrl_bounds(self) -> tuple[b2.Quantity, b2.Quantity]:
        """t of start and end of control period (including pre-trial control).
        Rounded to sample period to fully include trial.
        Just for the current trial."""
        if self._measuring_latency:
            return (0 * ms, 1 * ms)
        i_trial_bounds = self.cleo_prep.fsnn.i_trial_boundaries[self.i_current_trial]
        t_trial_bounds = [i * self.base_dt for i in i_trial_bounds]
        # control during trial and a little before
        t_ctrl_bounds = [
            t_trial_bounds[0] - self.ctrl_horizon,
            t_trial_bounds[1],
        ]
        # round to nearest ms so modulo will work as expected (not get 4.999999 for example)
        t_ctrl_bounds = [round(t / ms) * ms for t in t_ctrl_bounds]
        # round down/up to nearest sample period to fully include trial
        t_ctrl_bounds[0] = (
            floor(t_ctrl_bounds[0] / self.sample_period)
        ) * self.sample_period
        t_ctrl_bounds[1] = (
            ceil(t_ctrl_bounds[1] / self.sample_period)
        ) * self.sample_period
        T_ctrl = t_ctrl_bounds[1] - t_ctrl_bounds[0]
        assert np.isclose(
            round(T_ctrl / self.sample_period), T_ctrl / self.sample_period
        ), f"{T_ctrl=} not multiple of {self.sample_period=}"
        return tuple(t_ctrl_bounds)

    def process(self, state_dict, t_samp):
        # update trial bounds if previous one is over
        # round t_samp to avoid floating point issues
        t_samp = round(t_samp / ms) * ms
        n_trials = len(self.cleo_prep.fsnn.i_trial_boundaries)
        if t_samp >= self.t_ctrl_bounds[1]:
            if self.i_current_trial < n_trials - 1:
                log.info(f"finished control trial {self.i_current_trial}")
                self.i_current_trial += 1
                assert t_samp < self.t_ctrl_bounds[1], (
                    f"{t_samp=} not less than {self.t_ctrl_bounds[1]=} after incrementing trial"
                )
            else:
                assert self.i_current_trial == n_trials - 1, (
                    f"{self.i_current_trial=} not last trial {n_trials - 1=}. {self.t_ctrl_bounds=}"
                )

        if self.cleo_prep.probe is not None:
            if self.signals in ["spikes", "both"]:
                t, i, counts = state_dict["Probe"]["mua"]
                self.spike_counts.append(counts)
            else:
                counts = np.zeros(0)

            if self.signals in ["lfp", "both"]:
                lfp = state_dict["Probe"]["lfp"]
            else:
                lfp = np.zeros(0)
            y = np.concatenate([counts, lfp])
            assert y.shape == (self.m,), f"{y.shape=} not equal to {(self.m,)=}"

        if self.state_estimation == "kalman":
            xhat, P = self.update_state(self.xpred, y, self.Ppred)
        elif self.state_estimation == "ground truth":
            # need to be careful to create an entirely new array; strange crash even for array(y_mon.y)
            # xhat = np.zeros(self.n)
            # xhat[:] = self.cleo_prep.fsnn.ng_y.y[:]
            # need to zscore
            xhat = self.zscore(self.cleo_prep.fsnn.ng_y.y[:], "x")

        self._Xhat.append(xhat)
        self._R.append(self.reference_zscored(t_samp))
        uopto = self.compute_control(xhat)
        u = np.concatenate((uopto, self.u_task_zscored(t_samp)))

        if self.state_estimation == "kalman":
            self.xpred, self.Ppred = self.predict_state(xhat, u, P)

        uopto = self.unzscore(u, "u")[: self.k_opto]
        uopto = np.clip(uopto, 0, self.u_max_mWmm2)

        return {"light array": uopto * mwatt / mm2}, t_samp + self.sample_latency()

    @property
    def latency_mean(self) -> b2.Quantity:
        if self.add_latency:
            return self.baseline_latency + self.compute_latency_mean
        else:
            return 0 * ms

    def sample_latency(self) -> b2.Quantity:
        if self.add_latency:
            sampled = self.latency_rng.normal(
                loc=self.compute_latency_mean / b2.ms,
                scale=self.compute_latency_std / b2.ms,
            )
            sampled = max(sampled, 0)
            return self.baseline_latency + sampled * ms
        else:
            return 0 * ms

    def u_task_zscored(self, t: Quantity) -> np.ndarray:
        """returns task inputs at time t, z-scored"""
        if self.k_task == 0:
            return np.empty((np.size(t), 0)).squeeze()
        u_task = self.u_task(t)
        u_task_mean = self.lds.u_mean[self.k_opto :]
        u_task_std = self.lds.u_std[self.k_opto :]
        return (u_task - u_task_mean) / u_task_std

    @property
    def base_dt(self) -> b2.Quantity:
        """Returns base simulation timestep"""
        return self.cleo_prep.fsnn.ng_main.clock.dt

    def t_to_i_ctrl(self, t: b2.Quantity) -> int:
        return int(round(t / self.sample_period))

    def t_to_i_sim(self, t: b2.Quantity) -> int:
        return int(round(t / self.base_dt))

    # def update_with_trial_info(
    #     self, Xref, Zref, Utask, nsteps_trial, condition, i_trial_bounds, nsteps_iti
    # ):
    #     """call at start of each trial to update reference and task inputs.

    #     WARNING: only needed when running per trial, not with run_trials()!"""
    # if self.ref_type == "Z":
    #     ref = self.zscore(Zref, "z")
    # elif self.ref_type == "X":
    #     ref = self.zscore(Xref, "x")
    # t_start = self.t_now
    # self.reference_zscored = lambda t: array_vals_by_t(
    #     ref, t - t_start, self.base_dt
    # )
    # self.u_task = lambda t: array_vals_by_t(Utask, t - t_start, self.base_dt)
    # self.t_trial_bounds = tuple(t_start + i * self.base_dt for i in i_trial_bounds)

    # control during trial and a little before
    # self.t_ctrl_bounds = [
    #     self.t_trial_bounds[0] - self.ctrl_horizon,
    #     self.t_trial_bounds[1],
    # ]
    # # round to nearest ms so modulo will work as expected (not get 4.999999 for example)
    # self.t_ctrl_bounds = [round(t / ms) * ms for t in self.t_ctrl_bounds]
    # # round down/up to nearest sample period to fully include trial
    # self.t_ctrl_bounds[0] -= self.t_ctrl_bounds[0] % self.sample_period
    # self.t_ctrl_bounds[1] += (
    #     self.sample_period - (self.t_ctrl_bounds[1] % self.sample_period)
    # ) % self.sample_period
    # T_ctrl = self.t_ctrl_bounds[1] - self.t_ctrl_bounds[0]
    # assert np.isclose(
    #     round(T_ctrl / self.sample_period), T_ctrl / self.sample_period
    # ), f"{T_ctrl=} not multiple of {self.sample_period=}"


for attrib in [
    "C",
    "R",
    "A",
    "B",
    "Q",
    "n",
    "m",
    "m_z",
    "k",
    "k_task",
    "k_opto",
    "B_opto",
    "B_task",
]:
    setattr(
        Controller,
        attrib,
        property(lambda self, attrib=attrib: getattr(self.lds, attrib)),
    )
