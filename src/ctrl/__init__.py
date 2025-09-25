import time
from abc import ABC, abstractmethod
from collections.abc import Callable

import cleo
import cvxpy as cp
import numpy as np
from attrs import define, field
from BRAID.BRAIDModel import BRAIDModel
from BRAID.tools.LSSM import LSSM
from brian2 import Quantity, mm2, ms, mwatt, second
from jaxtyping import Float


@define
class Controller(cleo.ioproc.LatencyIOProcessor):
    k_opto: int = None
    """number of optogenetic inputs"""
    k_task: int = None
    """number of task-related inputs"""
    m: int = None
    """number of measured outputs (e.g. spike channels)"""
    rng: np.random.Generator = None
    sim: cleo.CLSimulator = None
    u_max_mWmm2: float = None
    add_latency: bool = True
    baseline_latency: Quantity = None
    braid_model: BRAIDModel = None
    lssm: LSSM = None
    compute_latency: Quantity = None
    reference: Callable[[Quantity], Quantity] = None
    u_task: Callable[[Quantity], np.ndarray] = field(default=lambda t: np.zeros(0))
    """callable with signature t -> task inputs (shape ([T], k_task) array)"""
    horizon: int = None
    """control horizon (number of time steps)"""
    Qcost: float = None
    Rcost: float = None

    spike_counts: list = field(factory=list)
    _Xhat: list = field(factory=list)
    _Xref: list = field(factory=list)
    xpred: np.ndarray = None
    Ppred: np.ndarray = None

    Uopt_opto: cp.Variable = None
    """optimized u trajectory over horizon"""
    Uopt_task: cp.Parameter = None
    """task inputs over horizon"""
    Xopt: cp.Variable = None
    """x trajectory over horizon including initial state"""
    Xopt_ref: cp.Parameter = None
    """reference trajectory over horizon"""
    x0opt: cp.Parameter = None
    """initial state parameter for optimization"""
    prob: cp.Problem = None
    """cvxpy optimization problem"""

    t_prev_trial_end: Quantity = 0 * ms
    t_trial_start: Quantity = 0 * ms
    t_trial_end: Quantity = 0 * ms

    def __attrs_post_init__(self):
        self.init_optimization()
        self.reset()

    # don't know why I need to make this a field to avoid attrs error
    _needs_off_schedule_sample: bool = False

    def reset(self):
        super().reset()
        self.xpred = np.zeros(self.nx)
        self.Ppred = np.eye(self.nx)  # initial covariance
        self.spike_counts = []
        self._Xhat = []
        self._Xref = []

        self.init_optimization()

    def init_optimization(self):
        self.Uopt_opto = cp.Variable((self.horizon, self.k_opto), nonneg=True)
        self.Uopt_task = cp.Parameter((self.horizon, self.k_task))
        self.Xopt = cp.Variable((self.horizon + 1, self.nx))
        self.Xopt_ref = cp.Parameter((self.horizon, self.nx))
        self.x0opt = cp.Parameter(self.nx)
        cost = (
            cp.sum_squares(self.Xopt[1:] - self.Xopt_ref) * self.Qcost / self.nx
            + cp.sum_squares(self.Uopt_opto) * self.Rcost / self.k_opto
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
            self.Uopt_opto <= self.u_max_zscored,
        ]
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

        self.Uopt_opto.value = np.zeros((self.horizon, self.k_opto))
        self.Xopt.value = np.zeros((self.horizon + 1, self.nx))

        assert self.prob.is_dcp(dpp=False), "problem not DCP"
        assert self.prob.is_dcp(dpp=True), "problem not DPP"

    def update_opt_problem(self, xhat, t_samp):
        self.x0opt.value = xhat
        self.Uopt_task.value = self.u_task_zscored(
            t_samp + np.arange(self.horizon) * self.sample_period
        )
        self.Xopt_ref.value = self.reference(
            self.t_now + np.arange(1, self.horizon + 1) * self.sample_period
        )

    @property
    def Xhat(self) -> Float[np.ndarray, "T nx"]:
        return np.array(self._Xhat)

    @property
    def Xref(self) -> Float[np.ndarray, "T nx"]:
        return np.array(self._Xref)

    @property
    def Yhat(self) -> Float[np.ndarray, "T m"]:
        return self.unzscore(self.Xhat @ self.C.T, "y")

    @property
    def k(self):
        return self.k_opto + self.k_task

    @property
    def u_max_zscored(self):
        return (
            self.u_max_mWmm2 - self.braid_model.uMean[: self.k_opto]
        ) / self.braid_model.uStd[: self.k_opto]

    @property
    def B_opto(self):
        return self.B[:, : self.k_opto]

    @property
    def B_task(self) -> Float[np.ndarray, "nx {self.k_task}"]:
        return self.B[:, self.k_opto :]

    @property
    def nx(self) -> int:
        return self.braid_model.nx

    @property
    def t_now(self) -> Quantity:
        return self.sim.network.t

    def measure_latency(self):
        durations = []
        xpred = np.zeros(self.nx)
        Ppred = np.eye(self.nx)
        u = self.rng.uniform(0, self.u_max_mWmm2, self.k) * mwatt / mm2
        u = self.rng.normal(self.braid_model.uMean, self.braid_model.uStd, self.k)
        for i in range(100):
            avg_rate = 20 / second
            y = self.rng.poisson(lam=avg_rate * self.sample_period)

            start = time.perf_counter()
            xhat, P = self.update_state(xpred, y, Ppred)
            uopto = self.compute_control(xhat)
            u = np.concatenate((uopto, self.u_task_zscored(10 * ms)))
            xpred, Ppred = self.predict_state(xhat, u, P)
            end = time.perf_counter()
            durations.append(end - start)

        # take off first 5 to let things warm up
        return np.array(durations)[5:].mean() * second

    def zscore(self, var: np.ndarray, varname: str):
        mean = getattr(self.braid_model, f"{varname}Mean")
        std = getattr(self.braid_model, f"{varname}Std")
        return (var - mean) / std

    def unzscore(self, var: np.ndarray, varname: str):
        mean = getattr(self.braid_model, f"{varname}Mean")
        std = getattr(self.braid_model, f"{varname}Std")
        return var * std + mean

    def update_state(self, xpred, y, Ppred) -> tuple[np.ndarray, np.ndarray]:
        """returns (xhat, P)"""
        assert np.linalg.norm(Ppred) < 1e100, "Ppred exploded"
        y = self.zscore(y, "y")
        yerr = y - self.C @ xpred  # innovation
        # use pinv instead of inv in case of singular matrix (where C has an all-0 zero)
        K = Ppred @ self.C.T @ np.linalg.pinv(self.C @ Ppred @ self.C.T + self.R)
        xhat = xpred + K @ yerr
        P = (np.eye(self.nx) - K @ self.C) @ Ppred
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
        t, i, counts = state_dict["Probe"]["mua"]
        self.spike_counts.append(counts)
        xhat, P = self.update_state(self.xpred, counts, self.Ppred)
        self._Xhat.append(xhat)
        self._Xref.append(self.reference(t_samp))
        uopto = self.compute_control(xhat)
        u = np.concatenate((uopto, self.u_task_zscored(t_samp)))
        self.xpred, self.Ppred = self.predict_state(xhat, u, P)

        uopto = self.unzscore(u, "u")[: self.k_opto]
        uopto = np.clip(uopto, 0, self.u_max_mWmm2)

        return {"light array": uopto * mwatt / mm2}, t_samp + self.latency

    @property
    def latency(self):
        if self.add_latency:
            return self.baseline_latency + self.compute_latency
        else:
            return 0 * ms

    def u_task_zscored(self, t: Quantity) -> np.ndarray:
        """returns task inputs at time t, z-scored"""
        if self.k_task == 0:
            return np.zeros(np.shape(t))
        u_task = self.u_task(t)
        u_task_mean = self.braid_model.uMean[self.k_opto :]
        u_task_std = self.braid_model.uStd[self.k_opto :]
        return (u_task - u_task_mean) / u_task_std

    def update_with_trial_info(
        self, t_prev_trial_end: Quantity, t_trial_start: Quantity, t_trial_end: Quantity
    ):
        """call at start of each trial to update reference and task inputs"""

        self.t_prev_trial_end = t_prev_trial_end
        self.t_trial_start = t_trial_start
        self.t_trial_end = t_trial_end


for attrib in ["C", "R", "A", "B", "Q"]:
    setattr(
        Controller,
        attrib,
        property(lambda self, attrib=attrib: getattr(self.lssm, attrib)),
    )
