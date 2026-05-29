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
    compute_latency_nonsolve: Quantity = None
    """computation not from QP solve itself: KF, canonicalization"""
    latency_rng: np.random.Generator = None
    lds: LDS = None
    """LDS or SubspaceLDS — both expose A, B, C, Q, R, x2z, etc."""
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
        # Nonlinear Cz (e.g. BRAID Cz1HL64U) cannot be constructed inside a
        # CVXPY problem — the MLP forward is not affine in CVXPY variables.
        # The PyTorch-based controllers (NLMPC, iLQR) handle it natively;
        # for the CVXPY-based controllers we linearize the nonlinear Cz at
        # x=0 (Jacobian + constant offset). The approximation is loose at
        # operating points far from origin, but keeps the comparison roughly
        # apples-to-apples on the same MLP topology — and far better than
        # falling back to X-tracking with a shape-mismatched Ropt.
        nonlinear_cz = (
            self.ref_type == "Z"
            and getattr(self.lds, 'cz_torch_module', None) is not None
        )
        if nonlinear_cz:
            import torch  # local: only the nonlinear-Cz path needs torch here
            cz_module = self.lds.cz_torch_module
            cz_module.eval()
            with torch.no_grad():
                x0_t = torch.zeros(self.n, dtype=torch.float32)
                cz0 = cz_module(x0_t).numpy()
            J0 = torch.autograd.functional.jacobian(cz_module, x0_t).numpy()
            log.warning(
                "%s: nonlinear Cz detected on LDS; linearizing at x=0 for "
                "CVXPY construction. ||J0||=%.3f, ||cz0||=%.3f. The Q_track "
                "metric will reflect the linearized approximation, not the "
                "full nonlinear Cz.",
                type(self).__name__, float(np.linalg.norm(J0)), float(np.linalg.norm(cz0)),
            )
            ctrl_var_traj = self.Xopt @ J0.T + cz0
        else:
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

    def measure_latency(self) -> tuple[b2.Quantity, b2.Quantity, b2.Quantity]:
        """returns mean, std, non-solve mean of control computation (with Brian unit)"""
        self._measuring_latency = True
        durations = []
        solve_times = []
        xpred = np.zeros(self.n)
        Ppred = np.eye(self.n)
        for i in range(100):
            avg_rate = 20 / second
            y = self.latency_rng.poisson(lam=avg_rate * self.sample_period)

            start = time.perf_counter()
            if self.state_estimation == "observer":
                xhat = xpred.copy()
                uopto = self.compute_control(xhat)
                u = np.concatenate((uopto, self.u_task_zscored(10 * ms)))
                n1 = self.lds.n1
                y_zscored = self.zscore(y, "y")
                x1_next, x2_next = self._observer_step(
                    self.lds, xpred[:n1], xpred[n1:], y_zscored, uopto,
                )
                xpred = np.concatenate([x1_next, x2_next])
            else:
                xhat, P = self.update_state(xpred, y, Ppred)
                uopto = self.compute_control(xhat)
                u = np.concatenate((uopto, self.u_task_zscored(10 * ms)))
                xpred, Ppred = self.predict_state(xhat, u, P)
            end = time.perf_counter()
            durations.append(end - start)
            solve_times.append(self.prob.solver_stats.solve_time)

        # take off first 5 to let things warm up
        # log.debug(f"latency measurement durations (s):\n{durations}")
        durations = np.array(durations[5:])
        solve_times = np.array(solve_times[5:])
        log.debug(f"{stats.describe(durations, bias=False)=}")
        log.debug(f"{stats.describe(solve_times, bias=False)=}")
        self._measuring_latency = False
        # to make robust to outliers from unreliable compute speed on cluster,
        # estimate mean and std using median and MAD
        median = np.median(durations)
        mad = np.median(np.abs(durations - median))
        # would include QP setup and Kalman filtering
        non_solve_time = np.median(np.subtract(durations, solve_times))
        return median * b2.second, mad / 0.6745 * b2.second, non_solve_time * b2.second

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
        K = np.linalg.solve(S, P_CT.T).T

        xhat = xpred + K @ yerr

        # standard P update is not very stable
        # P = (np.eye(self.n) - K @ self.C) @ Ppred
        # Joseph Form Update for P (Stability)
        IKC = np.eye(self.n) - K @ self.C
        P = IKC @ Ppred @ IKC.T + K @ self.R @ K.T
        # Force Symmetry
        P = (P + P.T) / 2

        return xhat, P

    def predict_state(self, xhat, u, P):
        """u is assumed to be z-scored. returns (xpred, Ppred)"""
        # P_pred = sim.A @ sim.P @ sim.A.T + sim.Q
        xpred = self.A @ xhat + self.B @ u
        Ppred = self.A @ P @ self.A.T + self.Q

        return xpred, Ppred

    @staticmethod
    def _observer_step(obs, x1, x2, y_zscored, u_opto_zscored):
        """Single USID observer step (predictor form).

        Computes x[t+1] from x[t], y[t], u[t] using the subspace observer
        equations directly, bypassing the Kalman filter.

        Args:
            obs: SubspaceLDS with A11, A22, K1, K2, C1, A12,
                 B1_prime, B2_prime attributes
            x1: (n1,) natural state at time t
            x2: (n2,) unnatural state at time t
            y_zscored: (ny,) z-scored observation at time t
            u_opto_zscored: (k_opto,) z-scored opto input at time t

        Returns:
            x1_next: (n1,) natural state at time t+1
            x2_next: (n2,) unnatural state at time t+1
        """
        innovation = y_zscored - obs.C1 @ x1
        x1_next = obs.A11 @ x1 + obs.K1 @ y_zscored + obs.B1_prime @ u_opto_zscored
        if obs.A12 is not None:
            x1_next += obs.A12 @ x2
        x2_next = obs.A22 @ x2 + obs.K2 @ innovation + obs.B2_prime @ u_opto_zscored
        return x1_next, x2_next

    def solve_optimization(self) -> None:
        """Solve the optimization problem with fallback through ordered solvers.

        When every solver fails (ill-posed relaxation for extreme k/m, etc.),
        fall back to zero control for this horizon rather than raising, so the
        fSNN simulation can continue and ctrl_data.pkl.gz gets written. The
        failure is surfaced via per-solver WARNINGs plus a final ERROR log.
        """
        for i, solver in enumerate(self.solvers):
            try:
                self.prob.solve(**solver.specific_args)

                if self.prob.status in [
                    "optimal",
                    "optimal_inaccurate",
                    "user_limit",
                ]:
                    if self.prob.status == "user_limit":
                        log.warning(
                            f"Solver {solver.name} reached user limit but found a feasible solution."
                        )
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

        log.error(
            f"All {len(self.solvers)} solvers failed. Last status: {self.prob.status}. "
            f"Falling back to zero control for this solve."
        )
        # Uopt_opto is z-scored; "no light" in raw space is zscore(0).
        # Uopt_task is a Parameter (inputs), not a Variable — don't touch.
        self.Uopt_opto.value = self.zscore(
            np.zeros((self.n_horizon, self.k)), "u"
        )[:, : self.k_opto]
        self.Xopt.value = np.zeros((self.n_horizon + 1, self.n))

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
        elif self.state_estimation == "observer":
            xhat = self.xpred.copy()
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
        elif self.state_estimation == "observer":
            n1 = self.lds.n1
            y_zscored = self.zscore(y, "y")
            x1_next, x2_next = self._observer_step(
                self.lds, self.xpred[:n1], self.xpred[n1:], y_zscored, uopto,
            )
            self.xpred = np.concatenate([x1_next, x2_next])

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
            time_limit_ms = self.solvers[0].generic_args.get("time_limit", 1e6) * 1000
            sampled = min(sampled, time_limit_ms)
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

    @property
    def R(self) -> Float[np.ndarray, "m m"]:
        """measurement noise covariance"""
        return np.reshape(self.lds.R, (self.m, self.m))

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
    # "R",  # defined above to reshape
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


