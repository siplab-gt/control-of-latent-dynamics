import time
from abc import abstractmethod
from collections.abc import Callable

import cleo
import cvxpy as cp
import numpy as np
from attrs import define, field
from brian2 import Quantity, mm2, ms, mwatt, second
from jaxtyping import Float

from src.model_fit import LDS
from src.scripts.cleo_prep import CleoPrep


@define
class Controller(cleo.ioproc.LatencyIOProcessor):
    rng: np.random.Generator = None
    sim: cleo.CLSimulator = None
    u_max_mWmm2: float = None
    add_latency: bool = True
    baseline_latency: Quantity = None
    compute_latency: Quantity = None
    lds: LDS = None
    reference: Callable[[Quantity], Quantity] = None
    u_task: Callable[[Quantity], np.ndarray] = field(default=lambda t: np.zeros(0))
    """callable with signature t -> task inputs (shape ([T], k_task) array)"""
    horizon: int = None
    """control horizon (number of time steps)"""
    Qcost: float = None
    Rcost: float = None
    cleo_prep: CleoPrep = None
    state_estimation: str = None

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
        self.xpred = np.zeros(self.n)
        self.Ppred = np.eye(self.n)  # initial covariance
        self.spike_counts = []
        self._Xhat = []
        self._Xref = []

        self.init_optimization()

    def init_optimization(self):
        self.Uopt_opto = cp.Variable((self.horizon, self.k_opto), nonneg=True)
        if self.k_task > 0:
            self.Uopt_task = cp.Parameter((self.horizon, self.k_task))
        self.Xopt = cp.Variable((self.horizon + 1, self.n))
        self.Xopt_ref = cp.Parameter((self.horizon, self.n))
        self.x0opt = cp.Parameter(self.n)
        cost = (
            cp.sum_squares(self.Xopt[1:] - self.Xopt_ref) * self.Qcost / self.n
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
        self.Xopt.value = np.zeros((self.horizon + 1, self.n))

        assert self.prob.is_dcp(dpp=False), "problem not DCP"
        assert self.prob.is_dcp(dpp=True), "problem not DPP"

    def update_opt_problem(self, xhat, t_samp):
        self.x0opt.value = xhat
        if self.k_task > 0:
            self.Uopt_task.value = self.u_task_zscored(
                t_samp + np.arange(self.horizon) * self.sample_period
            )
        self.Xopt_ref.value = self.reference(
            self.t_now + np.arange(1, self.horizon + 1) * self.sample_period
        )

    @property
    def Xhat(self) -> Float[np.ndarray, "T n"]:
        return np.array(self._Xhat)

    @property
    def Xref(self) -> Float[np.ndarray, "T n"]:
        return np.array(self._Xref)

    @property
    def Yhat(self) -> Float[np.ndarray, "T m"]:
        return self.unzscore(self.Xhat @ self.C.T, "y")

    @property
    def t_now(self) -> Quantity:
        return self.sim.network.t

    def measure_latency(self):
        durations = []
        xpred = np.zeros(self.n)
        Ppred = np.eye(self.n)
        u = self.rng.normal(self.lds.u_mean, self.lds.u_std, self.k)
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
            xhat = self.cleo_prep.fsnn.ng_y.y

        self._Xhat.append(xhat)
        self._Xref.append(self.reference(t_samp))
        uopto = self.compute_control(xhat)
        u = np.concatenate((uopto, self.u_task_zscored(t_samp)))

        if self.state_estimation == "kalman":
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
            return np.empty((np.size(t), 0)).squeeze()
        u_task = self.u_task(t)
        u_task_mean = self.lds.u_mean[self.k_opto :]
        u_task_std = self.lds.u_std[self.k_opto :]
        return (u_task - u_task_mean) / u_task_std

    @property
    def u_max_zscored(self):
        return (self.u_max_mWmm2 - self.lds.u_mean[: self.k_opto]) / self.lds.u_std[
            : self.k_opto
        ]

    def update_with_trial_info(
        self, t_prev_trial_end: Quantity, t_trial_start: Quantity, t_trial_end: Quantity
    ):
        """call at start of each trial to update reference and task inputs"""

        self.t_prev_trial_end = t_prev_trial_end
        self.t_trial_start = t_trial_start
        self.t_trial_end = t_trial_end


for attrib in [
    "C",
    "R",
    "A",
    "B",
    "Q",
    "n",
    "m",
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
