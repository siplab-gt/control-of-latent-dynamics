import time
from abc import abstractmethod
from collections.abc import Callable

import brian2.only as b2
import cleo
import cvxpy as cp
import numpy as np
from attrs import define, field
from brian2 import Quantity, mm2, ms, mwatt, second
from jaxtyping import Float

from src.helpers import array_vals_by_t
from src.model_fit import LDS
from src.scripts.cleo_prep import CleoPrep


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
    reference_zscored: Callable[[Quantity], Quantity] = None
    u_task: Callable[[Quantity], np.ndarray] = field(default=lambda t: np.zeros(0))
    """callable with signature t -> task inputs (shape ([T], k_task) array)"""
    ctrl_horizon: Quantity = None
    """control horizon (with time unit). In OL case, this is pre-trial control time."""
    Qcost: float = None
    Rcost: float = None
    cleo_prep: CleoPrep = None
    state_estimation: str = None

    spike_counts: list = field(factory=list)
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

    t_trial_bounds: tuple[Quantity, Quantity] = field(
        factory=lambda: (0 * ms, 1000 * ms)
    )
    """t of start and end of trial (not ITI) where we want to minimize cost"""
    t_ctrl_bounds: list = field(factory=lambda: [0 * ms, 1 * ms])
    """t of start and end of control period (including pre-trial control).
    Rounded to sample period to fully include trial."""

    def __attrs_post_init__(self):
        self.reference_zscored = lambda t: np.zeros((*np.shape(t), self.n_r))
        self.reset()

    # don't know why I need to make this a field to avoid attrs error
    _needs_off_schedule_sample: bool = False

    def reset(self):
        super().reset()
        self.xpred = np.zeros(self.n)
        self.Ppred = np.eye(self.n)  # initial covariance
        self.spike_counts = []
        self._Xhat = []
        self._R = []

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
        return self.sim.network.t

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
        durations = np.array(durations[5:])
        return durations.mean() * b2.second, durations.std() * b2.second

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
        K = Ppred @ self.C.T @ np.linalg.pinv(self.C @ Ppred @ self.C.T + self.R)
        xhat = xpred + K @ yerr
        P = (np.eye(self.n) - K @ self.C) @ Ppred
        return xhat, P

    def predict_state(self, xhat, u, P):
        """u is assumed to be z-scored. returns (xpred, Ppred)"""
        # P_pred = sim.A @ sim.P @ sim.A.T + sim.Q
        xpred = self.A @ xhat + self.B @ u
        Ppred = self.A @ P @ self.A.T + self.Q

        return xpred, Ppred

    @abstractmethod
    def compute_control(self, xhat):
        pass

    def process(self, state_dict, t_samp):
        if self.cleo_prep.probe is not None:
            t, i, counts = state_dict["Probe"]["mua"]
        else:
            counts = np.zeros(0, dtype=np.uint32)
        self.spike_counts.append(counts)

        if self.state_estimation == "kalman":
            xhat, P = self.update_state(self.xpred, counts, self.Ppred)
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

    def update_with_trial_info(
        self, Xref, Zref, Utask, nsteps_trial, condition, i_trial_bounds, nsteps_iti
    ):
        """call at start of each trial to update reference and task inputs"""
        if self.ref_type == "Z":
            ref = self.zscore(Zref, "z")
        elif self.ref_type == "X":
            ref = self.zscore(Xref, "x")
        t_start = self.t_now
        self.reference_zscored = lambda t: array_vals_by_t(
            ref, t - t_start, self.base_dt
        )
        self.u_task = lambda t: array_vals_by_t(Utask, t - t_start, self.base_dt)
        self.t_trial_bounds = tuple(t_start + i * self.base_dt for i in i_trial_bounds)

        # control during trial and a little before
        self.t_ctrl_bounds = [
            self.t_trial_bounds[0] - self.ctrl_horizon,
            self.t_trial_bounds[1],
        ]
        # round down/up to nearest sample period to fully include trial
        self.t_ctrl_bounds[0] -= self.t_ctrl_bounds[0] % self.sample_period
        self.t_ctrl_bounds[1] += (
            self.sample_period - (self.t_ctrl_bounds[1] % self.sample_period)
        ) % self.sample_period
        T_ctrl = self.t_ctrl_bounds[1] - self.t_ctrl_bounds[0]
        assert np.isclose(
            round(T_ctrl / self.sample_period), T_ctrl / self.sample_period
        ), f"{T_ctrl=} not multiple of {self.sample_period=}"


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
