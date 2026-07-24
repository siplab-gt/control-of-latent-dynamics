"""
Nonlinear MPC controller using a differentiable forward model.

Optimizes light inputs directly through the opsin model + linear state
dynamics, replacing the CVXPY QP with:

    light → opsin GRU → photocurrents → B' @ p → state update

The inner solver is box-constrained L-BFGS-B by default (scipy, quasi-Newton
with native [0, u_max] bounds and a robust gradient-norm convergence test):
it converges in far fewer function evaluations than first-order Adam on this
smooth objective, and its built-in tolerance test avoids the flat-tail
oscillation that made Adam's relative-improvement early-stop fragile.
Projected-gradient Adam is retained as the 'adam' option. Both warm-start from
the previous solution (shifted by one step).
"""
import logging
import time

import brian2.only as b2
import numpy as np
import scipy.optimize as sopt
import torch
from attrs import define, field
from brian2 import mm2, ms, mwatt

from src.ctrl import Controller
from src.ctrl._cz_helpers import apply_cz, build_cz_state, has_cz

log = logging.getLogger(__name__)


def _rollout_state(x0, u_for_B, A_cl, B_block, ref_trajectory, ref_type,
                   cz_state=None):
    """Roll out the closed-loop subspace-LDS state through the MPC horizon.

    The observer-form update
        x1' = A11 x1 + K1 (C1 x1 + C2 x2) + B1' u (+ A12 x2)
        x2' = A22 x2 + K2 (C2 x2) + B2' u
    is exactly linear in x = [x1; x2] (the innovation y_pred - C1 x1 equals
    C2 x2), so it is a single closed-loop block system
        x_{t+1} = A_cl x_t + B_block u_t,
        A_cl = [[A11 + K1 C1, K1 C2 + A12], [0, A22 + K2 C2]],
        B_block = [B1'; B2'].
    A_cl / B_block are precomputed once from the frozen model in
    `_setup_torch_matrices`; here the driven term B_block u is applied to the
    whole horizon in one matmul and only the small A_cl x recurrence is serial.

    Args:
        x0: (n,) initial state [x1; x2] (torch tensor)
        u_for_B: (H, k_opto) input B' is multiplied by — `photocurrents`
            (NN_opsin output) in opsin-mode, `u_zscored` in linearized-subspace
            mode (USID `use_opsin=False`). Same recurrence; caller picks.
        A_cl: (n, n) closed-loop state matrix (torch tensor)
        B_block: (n, k_opto) stacked [B1'; B2'] input matrix (torch tensor)
        ref_trajectory: (H, n_r) reference in z-scored space (torch tensor)
        ref_type: "X" or "Z"
        cz_state: dict from `build_cz_state` describing linear/nonlinear
            Cz dispatch. Required if ref_type='Z'; ignored otherwise.

    Returns:
        tracking_cost: scalar (torch tensor)
    """
    H = u_for_B.shape[0]
    driven = u_for_B @ B_block.T  # (H, n): B_block u_t for the whole horizon

    x = x0
    cost = torch.tensor(0.0, dtype=x0.dtype, device=x0.device)
    for t in range(H):
        x = A_cl @ x + driven[t]
        if ref_type == "Z" and cz_state is not None and has_cz(cz_state):
            ctrl_var = apply_cz(cz_state, x)
        else:
            ctrl_var = x
        cost = cost + torch.sum((ctrl_var - ref_trajectory[t]) ** 2)

    return cost / H


def _rollout_state_unified(x0, u_zscored, A, B_opto, ref_trajectory, ref_type,
                           cz_state=None):
    """Roll out unified-LDS dynamics for BRAID-shaped models.

    Open-loop (no observer correction in the rollout): BRAID's LDS doesn't
    expose a Kalman gain in the form K1/K2 that the subspace observer uses.
    The kalman update happens at `process()` time using the base-class
    `update_state` with the actual y observation; the planning rollout
    here is a deterministic forward sim of `x[t+1] = A·x + B_opto·u_z`.

    Args:
        x0: (n,) initial state (torch tensor)
        u_zscored: (H, k_opto) z-scored opto input (torch tensor)
        A: (n, n) unified state matrix (torch tensor)
        B_opto: (n, k_opto) unified opto input matrix, trained against
            z-scored light (torch tensor)
        ref_trajectory: (H, n_r) reference in z-scored space (torch tensor)
        ref_type: "X" or "Z"
        Cz: (n_r, n) behavioral decoder (torch tensor), needed if ref_type="Z"

    Returns:
        tracking_cost: scalar (torch tensor)
    """
    H = u_zscored.shape[0]
    x = x0
    cost = torch.tensor(0.0, dtype=x0.dtype, device=x0.device)

    for t in range(H):
        x = A @ x + B_opto @ u_zscored[t]

        if ref_type == "Z" and cz_state is not None and has_cz(cz_state):
            ctrl_var = apply_cz(cz_state, x)
        else:
            ctrl_var = x
        cost = cost + torch.sum((ctrl_var - ref_trajectory[t]) ** 2)

    return cost / H


def _solve_projected_adam(loss_fn, u0, lr, max_iters, tol, patience, u_max):
    """Projected-gradient Adam for the MPC light optimization.

    Minimizes ``loss_fn(u)`` over the box ``[0, u_max]`` (clamped after each
    Adam step) and stops early once the relative loss improvement stays below
    ``tol`` for ``patience`` consecutive iterations, else runs ``max_iters``.

    The previous fixed 5–10 iteration loop did not converge. On the linear
    (no-opsin) model the MPC objective is convex and identical — up to the
    constant 1/H factor — to LQMPC's QP, so an un-converged solve was the sole
    reason NLMPC under-performed LQMPC and over-drove the light: with too few
    iterations the optimizer never reached the shared optimum. The convergence
    check makes the iteration count adaptive (cheap on the convex linear path,
    bounded on the opsin path), and is backward-compatible — with the old small
    ``max_iters`` the patience window never closes, so behavior is unchanged.

    Returns ``(u_opt: np.ndarray, n_iters: int, final_loss: float)``.
    """
    u = torch.tensor(u0, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=lr)
    prev = None
    n_converged = 0
    n_iters = 0
    final_loss = float("nan")
    for i in range(int(max_iters)):
        optimizer.zero_grad()
        loss = loss_fn(u)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            u.clamp_(0.0, u_max)
        n_iters = i + 1
        cur = float(loss.detach())
        final_loss = cur
        if prev is not None:
            rel = abs(prev - cur) / max(abs(prev), 1e-12)
            if rel < tol:
                n_converged += 1
                if n_converged >= patience:
                    break
            else:
                n_converged = 0
        prev = cur
    return u.detach().numpy().copy(), n_iters, final_loss


def _solve_lbfgsb(loss_fn, u0, u_max, max_iters, ftol, gtol):
    """Box-constrained L-BFGS-B for the MPC light optimization (default).

    Minimizes ``loss_fn(u)`` over the box ``[0, u_max]`` with scipy's L-BFGS-B
    — a quasi-Newton method that handles the bounds natively and stops on a
    proper gradient-norm / relative-function test (``gtol`` / ``ftol``). The
    differentiable torch ``loss_fn`` supplies value+gradient via autograd.

    This replaces first-order Adam as the inner solver: on this smooth
    (GRU + linear) objective the curvature information converges in far fewer
    function evaluations, which (a) makes the non-convex USID+opsin solve
    feasibly convergeable — Adam needed ~1.6k cold-start iters — and (b) is
    robust to the flat-tail per-iter oscillation that broke Adam's
    consecutive-relative-improvement early-stop. The optimum is unchanged
    (same objective); only the path/speed differ — for the convex linear path
    it reaches the identical LQMPC/QP solution.

    ``max_iters`` caps L-BFGS-B iterations; func evals are capped at 4×.
    Returns ``(u_opt: np.ndarray float32, n_iters: int, final_loss: float)``.
    """
    shape = u0.shape

    def f_and_g(u_flat):
        u = torch.tensor(
            u_flat.reshape(shape), dtype=torch.float32, requires_grad=True
        )
        loss = loss_fn(u)
        loss.backward()
        return float(loss.detach()), u.grad.detach().numpy().ravel().astype(np.float64)

    res = sopt.minimize(
        f_and_g,
        u0.ravel().astype(np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, float(u_max))] * int(u0.size),
        options={
            "maxiter": int(max_iters),
            "maxfun": int(max_iters) * 4,
            "ftol": ftol,
            "gtol": gtol,
        },
    )
    return res.x.reshape(shape).astype(np.float32), int(res.nit), float(res.fun)


@define
class NLMPCController(Controller):
    """Nonlinear MPC using PyTorch gradient descent.

    Three rollout modes are supported, auto-detected from the LDS shape:

      1. **opsin_subspace** — USID with `use_opsin=True`. LDS has subspace
         decomposition (A11/A22/K1/K2/B1'/B2'/Cz) and an attached NN opsin.
         Rollout: light → NN_opsin (with GRU memory) → photocurrent → B'·p
         → state. State estimation: observer (auto-selected via `K1`).
      2. **linearized_subspace** — USID with `use_opsin=False`. Same
         subspace LDS, but no opsin model. B' was trained against z-scored
         light directly. Rollout: light → z-score → B'·u_z → state.
         State estimation: observer.
      3. **linearized_unified** — BRAID. LDS has unified A/B_opto/C/Cz, no
         subspace decomposition, no opsin. Rollout: light → z-score → A·x
         + B_opto·u_z. State estimation: kalman (auto-selected via lack
         of K1 in `ctrl_expts.py`).

    Mode flags `_subspace_mode` and `_has_opsin` are populated in
    `_setup_torch_matrices` from the LDS attribute set.
    """
    nlmpc_grad_iters: int = 10
    """Maximum projected-Adam iterations per control step (the convergence
    check stops earlier when the loss plateaus). Wired from the `nlmpc_iters`
    wildcard. Must be large enough for the solve to converge — the old default
    of 5–10 left the optimizer un-converged."""
    nlmpc_step_size: float = 0.5
    nlmpc_tol: float = 5e-5
    """Relative-loss improvement threshold for early-stop convergence. Set to
    5e-5 (not 1e-6): at high input dim (k=32) Adam's flat-tail per-iter
    improvement oscillates ~1e-5, so a 1e-6 threshold never closes the patience
    window and the solve runs the full cap — yet the solution is already
    converged (<0.03% of a 5x-longer solve) by then. 5e-5 lets early-stop fire
    in the converged tail; it cannot fire prematurely (only once the loss is
    flat) and is bounded above by nlmpc_grad_iters."""
    nlmpc_patience: int = 5
    """Consecutive sub-`tol` iterations required to declare convergence."""
    nlmpc_optimizer: str = "lbfgsb"
    """Inner solver: 'lbfgsb' (box-constrained scipy L-BFGS-B — default;
    quasi-Newton, converges in far fewer func-evals than Adam and has a robust
    gradient-norm convergence test) or 'adam' (legacy projected-gradient Adam,
    kept for comparison; uses nlmpc_step_size / nlmpc_tol / nlmpc_patience)."""
    nlmpc_ftol: float = 1e-9
    """L-BFGS-B relative-function convergence tolerance."""
    nlmpc_gtol: float = 1e-6
    """L-BFGS-B projected-gradient inf-norm convergence tolerance."""

    # Internal state (not constructor args)
    _u_light_warm: np.ndarray = field(default=None, init=False, repr=False)
    _torch_matrices: dict = field(factory=dict, init=False, repr=False)
    _opsin_model: object = field(default=None, init=False, repr=False)
    _opsin_hidden: object = field(default=None, init=False, repr=False)
    # Mode flags: detected from LDS shape in _setup_torch_matrices.
    _subspace_mode: bool = field(default=False, init=False, repr=False)
    _has_opsin: bool = field(default=False, init=False, repr=False)
    # Cz dispatch state — populated by _cz_helpers.build_cz_state.
    _cz_state: dict = field(factory=dict, init=False, repr=False)
    # Per-cell history of solve iteration counts, for log visibility (how many
    # Adam iters the convergence check actually used per control step).
    _n_iters_history: list = field(factory=list, init=False, repr=False)

    def sample_latency(self) -> b2.Quantity:
        """Sample latency without referencing CVXPY solvers."""
        if self.add_latency:
            sampled = self.latency_rng.normal(
                loc=self.compute_latency_mean / b2.ms,
                scale=self.compute_latency_std / b2.ms,
            )
            sampled = max(sampled, 0)
            return self.baseline_latency + sampled * ms
        else:
            return 0 * ms

    def reset(self):
        super().reset()
        self._u_light_warm = None
        self._opsin_hidden = None
        self._n_iters_history = []
        self._setup_torch_matrices()

    def _setup_torch_matrices(self):
        """Extract state-space matrices from LDS, set mode flags, and
        convert to torch tensors. Handles three LDS shapes — see class
        docstring."""
        lds = self.lds
        self._subspace_mode = hasattr(lds, 'A11')
        self._has_opsin = (
            hasattr(lds, 'opsin_model') and lds.opsin_model is not None
        )

        if self._subspace_mode:
            self._torch_matrices = {
                'A11': torch.tensor(lds.A11, dtype=torch.float32),
                'A22': torch.tensor(lds.A22, dtype=torch.float32),
                'K1': torch.tensor(lds.K1, dtype=torch.float32),
                'K2': torch.tensor(lds.K2, dtype=torch.float32),
                'C1': torch.tensor(lds.C1, dtype=torch.float32),
                'C2': torch.tensor(lds.C2, dtype=torch.float32),
                'B1_prime': torch.tensor(lds.B1_prime, dtype=torch.float32),
                'B2_prime': torch.tensor(lds.B2_prime, dtype=torch.float32),
                'A12': torch.tensor(lds.A12, dtype=torch.float32) if lds.A12 is not None else None,
            }
            # Precompute the closed-loop block system the rollout iterates:
            # x_{t+1} = A_cl x_t + B_block u_t (frozen model → build once). See
            # _rollout_state for the derivation (innovation = C2 x2).
            m = self._torch_matrices
            n1 = m['A11'].shape[0]
            n2 = m['A22'].shape[0]
            A_cl = torch.zeros((n1 + n2, n1 + n2), dtype=torch.float32)
            A_cl[:n1, :n1] = m['A11'] + m['K1'] @ m['C1']
            top_right = m['K1'] @ m['C2']
            if m['A12'] is not None:
                top_right = top_right + m['A12']
            A_cl[:n1, n1:] = top_right
            A_cl[n1:, n1:] = m['A22'] + m['K2'] @ m['C2']
            m['A_cl'] = A_cl
            m['B_block'] = torch.cat([m['B1_prime'], m['B2_prime']], dim=0)
        else:
            # Unified-LDS shape (BRAID). Only A / B_opto / C / Cz are present.
            self._torch_matrices = {
                'A': torch.tensor(lds.A, dtype=torch.float32),
                'B_opto': torch.tensor(lds.B_opto, dtype=torch.float32),
                'C': torch.tensor(lds.C, dtype=torch.float32),
            }
        # Single source of truth for Cz dispatch across subspace and unified
        # modes, linear and nonlinear. See src/ctrl/_cz_helpers.py.
        self._cz_state = build_cz_state(lds)

        if self._has_opsin:
            self._opsin_model = lds.opsin_model
            self._opsin_model.eval()
            for p in self._opsin_model.parameters():
                p.requires_grad_(False)
        else:
            # Pre-compute z-scoring tensors for the linearized rollouts.
            # B' (subspace) and B (unified) under no-opsin were trained
            # against z-scored light, so we must z-score u_light before
            # multiplying. Using torch tensors avoids per-iter numpy ↔ torch
            # round-trips inside the Adam loop.
            self._torch_matrices['u_mean_opto'] = torch.tensor(
                lds.u_mean[: self.k_opto], dtype=torch.float32
            )
            self._torch_matrices['u_std_opto'] = torch.tensor(
                lds.u_std[: self.k_opto], dtype=torch.float32
            )

    def _run_opsin_forward(self, u_light, h0=None):
        """Run opsin model: (H, k_opto) light → (H, k_opto) photocurrents.

        u_light enters the NN per-channel; B' (learned at model_fit time)
        maps the per-channel photocurrent output back into state.

        ARCHITECTURAL INVARIANT: NN output is per-channel and approximates
        the average dynamics of that channel's cluster. Per-neuron output
        is NOT a goal of this model. Do not introduce per-neuron output
        dimensions — breaks the B' contract (B' is dimensioned
        [n_state, n_channels]).

        Always uses the stateful path (`forward_with_hidden`) so the GRU
        hidden state bootstraps on first call (passing `h0=None` lets
        `nn.GRU` initialise to zeros internally and return a valid `h_out`
        the caller can persist). The earlier non-stateful fallback dropped
        `h_out`, which silently kept `_opsin_hidden = None` forever — the
        controller's opsin model then ran without memory of the
        photocurrent decaying in tissue.

        Args:
            u_light: (H, k_opto) light input tensor.
            h0: Optional GRU hidden state from previous timestep.
                Shape: (num_layers, k_opto, hidden_size).

        Returns:
            photocurrents: (H, k_opto) tensor.
            h_out: GRU hidden state after processing.
        """
        u_flat = u_light.T.unsqueeze(-1)  # (k_opto, H, 1)
        p_flat, h_out = self._opsin_model.forward_with_hidden(u_flat, h0)
        return p_flat.T, h_out

    def compute_control(self, xhat):
        if self.t_now < self.t_ctrl_bounds[0] or self.t_now >= self.t_ctrl_bounds[1]:
            return np.zeros(self.k_opto)

        H = self.n_horizon
        k_opto = self.k_opto

        # Warm-start: shift previous solution left by 1, duplicate last step
        if self._u_light_warm is not None and self._u_light_warm.shape[0] == H:
            warm = np.roll(self._u_light_warm, -1, axis=0)
            warm[-1] = warm[-2] if H > 1 else 0.0
        else:
            warm = np.zeros((H, k_opto))

        x0 = torch.tensor(xhat, dtype=torch.float32)

        # Reference trajectory over horizon
        t_horizon = self.t_now + np.arange(1, H + 1) * self.sample_period
        ref = torch.tensor(self.reference_zscored(t_horizon), dtype=torch.float32)

        mat = self._torch_matrices

        # Detach hidden state for MPC planning: all iterations start from the
        # same opsin state (the real current state), not from each other.
        # Only meaningful in opsin mode.
        h0_planning = (
            self._opsin_hidden.detach() if self._opsin_hidden is not None else None
        )

        def loss_fn(u_light):
            # Mode-aware rollout. Three paths:
            #   • opsin + subspace: forward through NN opsin → photocurrents,
            #     then subspace observer-form recurrence with B'·photocurrent.
            #   • no-opsin + subspace: skip NN opsin, z-score u_light directly,
            #     same subspace observer-form recurrence with B'·u_zscored
            #     (B' was trained against z-scored light when use_opsin=False).
            #   • no-opsin + unified (BRAID): unified A·x + B_opto·u_zscored,
            #     no observer correction (BRAID has no K1/K2 gain in this
            #     form; kalman update happens out-of-band in `process`).
            if self._has_opsin:
                photocurrents, _ = self._run_opsin_forward(u_light, h0=h0_planning)
                tracking_cost = _rollout_state(
                    x0, photocurrents,
                    mat['A_cl'], mat['B_block'],
                    ref, self.ref_type, self._cz_state,
                )
            elif self._subspace_mode:
                u_zscored = (u_light - mat['u_mean_opto']) / mat['u_std_opto']
                tracking_cost = _rollout_state(
                    x0, u_zscored,
                    mat['A_cl'], mat['B_block'],
                    ref, self.ref_type, self._cz_state,
                )
            else:  # linearized_unified (BRAID)
                u_zscored = (u_light - mat['u_mean_opto']) / mat['u_std_opto']
                tracking_cost = _rollout_state_unified(
                    x0, u_zscored,
                    mat['A'], mat['B_opto'],
                    ref, self.ref_type, self._cz_state,
                )
            # Control effort cost (penalizes light power). Tracking + effort is
            # LQMPC's exact QP objective up to the constant 1/H factor (same
            # argmin) — so on the linear no-opsin path a converged solve must
            # match LQMPC. See _solve_projected_adam.
            effort_cost = torch.sum(u_light ** 2) / (H * k_opto)
            return self.Qcost * tracking_cost / self.n_r + self.Rcost * effort_cost

        if self.nlmpc_optimizer == "lbfgsb":
            u_opt, n_iters, final_loss = _solve_lbfgsb(
                loss_fn, warm,
                u_max=self.u_max_mWmm2,
                max_iters=self.nlmpc_grad_iters,
                ftol=self.nlmpc_ftol,
                gtol=self.nlmpc_gtol,
            )
        else:
            u_opt, n_iters, final_loss = _solve_projected_adam(
                loss_fn, warm,
                lr=self.nlmpc_step_size,
                max_iters=self.nlmpc_grad_iters,
                tol=self.nlmpc_tol,
                patience=self.nlmpc_patience,
                u_max=self.u_max_mWmm2,
            )
        if not self._measuring_latency:
            self._n_iters_history.append(n_iters)
            # Visible (INFO) cumulative summary every 100 control steps so the
            # actual iters the convergence check used are greppable per cell
            # ("NLMPC iters used"); shows whether early-stop fires (median < cap)
            # or the solve hits the cap (at_cap high).
            if len(self._n_iters_history) % 100 == 0:
                h = np.asarray(self._n_iters_history)
                log.info(
                    "NLMPC iters used: n=%d mean=%.0f median=%.0f min=%d max=%d "
                    "at_cap(%d)=%.0f%% (last final_loss %.6g)",
                    len(h), h.mean(), np.median(h), int(h.min()), int(h.max()),
                    self.nlmpc_grad_iters,
                    100.0 * (h >= self.nlmpc_grad_iters).mean(), final_loss,
                )

        # Store for warm-starting next step
        self._u_light_warm = u_opt

        # Return first step's light command (raw mW/mm²)
        return self._u_light_warm[0]

    def process(self, state_dict, t_samp):
        """Override process to handle opsin-aware observer and raw light output."""
        t_samp = round(t_samp / ms) * ms
        n_trials = len(self.cleo_prep.fsnn.i_trial_boundaries)
        if t_samp >= self.t_ctrl_bounds[1]:
            if self.i_current_trial < n_trials - 1:
                log.info(f"finished control trial {self.i_current_trial}")
                self.i_current_trial += 1
                assert t_samp < self.t_ctrl_bounds[1], (
                    f"{t_samp=} not less than {self.t_ctrl_bounds[1]=} after incrementing trial"
                )
                # Trial boundary: clear NLMPC per-trial state so warm-start
                # light and opsin GRU memory don't leak into the next trial.
                # Each trial has a different reference and starts with the
                # opsin in tissue at rest (whatever ITI decay produced).
                # This branch fires exactly once per trial transition (the
                # outer `t_samp >= t_ctrl_bounds[1]` check requires t_samp
                # to have just crossed the prior trial's window end), so
                # within-trial control steps keep their warm-start intact.
                self._u_light_warm = None
                self._opsin_hidden = None
            else:
                assert self.i_current_trial == n_trials - 1

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

        # State estimation
        if self.state_estimation == "kalman":
            xhat, P = self.update_state(self.xpred, y, self.Ppred)
        elif self.state_estimation == "observer":
            xhat = self.xpred.copy()
        elif self.state_estimation == "ground truth":
            xhat = self.zscore(self.cleo_prep.fsnn.ng_y.y[:], "x")

        self._Xhat.append(xhat)
        self._R.append(self.reference_zscored(t_samp))

        # Compute optimal light (raw mW/mm²)
        u_light = self.compute_control(xhat)
        u_light = np.clip(u_light, 0, self.u_max_mWmm2)

        # Observer / kalman update — mode-aware. Three paths mirror the
        # rollout structure in `compute_control`:
        if self.state_estimation == "observer" and self._has_opsin:
            # opsin_subspace: run NN opsin on applied light to advance the
            # GRU state by one real timestep, feed photocurrent to the
            # subspace observer step (B' was trained on photocurrent).
            n1 = self.lds.n1
            y_zscored = self._observer_y_zscored(y)
            with torch.no_grad():
                light_tensor = torch.tensor(
                    u_light.reshape(1, -1), dtype=torch.float32
                )
                p_tensor, h_out = self._run_opsin_forward(
                    light_tensor, h0=self._opsin_hidden
                )
                self._opsin_hidden = h_out
                p_current = p_tensor[0].numpy()
            x1_next, x2_next = self._observer_step(
                self.lds, self.xpred[:n1], self.xpred[n1:], y_zscored, p_current,
            )
            self.xpred = np.concatenate([x1_next, x2_next])
        elif self.state_estimation == "observer" and not self._has_opsin:
            # linearized_subspace (USID, use_opsin=False): no opsin, but
            # subspace observer step still applies. B' was trained on
            # z-scored light directly under `use_opsin=False`, so feed
            # u_zscored to the same `_observer_step` slot.
            n1 = self.lds.n1
            y_zscored = self._observer_y_zscored(y)
            u_opto_zscored = self.zscore(
                np.concatenate([u_light, np.zeros(self.k_task)]), "u"
            )[: self.k_opto]
            x1_next, x2_next = self._observer_step(
                self.lds, self.xpred[:n1], self.xpred[n1:], y_zscored, u_opto_zscored,
            )
            self.xpred = np.concatenate([x1_next, x2_next])
        elif self.state_estimation == "kalman":
            # linearized_unified (BRAID): mirror the base-class
            # `Controller.process` u-construction (lines 453-457) but
            # z-score the opto portion ourselves since NLMPC's
            # compute_control returns raw mW/mm² (LQMPC returns z-scored
            # already). Index lds.u_mean / u_std at [:k_opto] directly
            # instead of zero-padding then slicing — the base class's
            # u_mean has shape (k_opto+k_task,) only when
            # model_task_inputs=True; otherwise it's (k_opto,) and the
            # earlier `np.concatenate([u_light, self.u_task(t_samp)])`
            # produced a shape mismatch against u_mean.
            u_opto_zscored = (
                u_light - self.lds.u_mean[: self.k_opto]
            ) / self.lds.u_std[: self.k_opto]
            u = np.concatenate([u_opto_zscored, self.u_task_zscored(t_samp)])
            self.xpred, self.Ppred = self.predict_state(xhat, u, P)

        return {"light array": u_light * mwatt / mm2}, t_samp + self.sample_latency()

    def measure_latency(self):
        """Measure computation latency for the nonlinear MPC.

        Mode-aware mirror of `process` so latency includes the dominant
        per-step cost in each mode. opsin-mode pays the GRU forward; the
        linearized modes don't.
        """
        self._measuring_latency = True
        durations = []
        xpred = np.zeros(self.n)
        opsin_hidden = None
        Ppred = np.eye(self.n) if not self._subspace_mode else None  # kalman state

        for i in range(100):
            avg_rate = 20 / b2.second
            y = self.latency_rng.poisson(lam=avg_rate * self.sample_period)

            start = time.perf_counter()

            if self.state_estimation == "observer" and self._has_opsin:
                xhat = xpred.copy()
                self._opsin_hidden = opsin_hidden
                u_light = self.compute_control(xhat)
                n1 = self.lds.n1
                y_zscored = self._observer_y_zscored(y)
                with torch.no_grad():
                    light_t = torch.tensor(
                        u_light.reshape(1, -1), dtype=torch.float32
                    )
                    p_t, h_out = self._run_opsin_forward(light_t, h0=opsin_hidden)
                    opsin_hidden = h_out
                    p_np = p_t[0].numpy()
                x1_next, x2_next = self._observer_step(
                    self.lds, xpred[:n1], xpred[n1:], y_zscored, p_np,
                )
                xpred = np.concatenate([x1_next, x2_next])
            elif self.state_estimation == "observer" and not self._has_opsin:
                xhat = xpred.copy()
                u_light = self.compute_control(xhat)
                n1 = self.lds.n1
                y_zscored = self._observer_y_zscored(y)
                u_opto_zscored = self.zscore(
                    np.concatenate([u_light, np.zeros(self.k_task)]), "u"
                )[: self.k_opto]
                x1_next, x2_next = self._observer_step(
                    self.lds, xpred[:n1], xpred[n1:], y_zscored, u_opto_zscored,
                )
                xpred = np.concatenate([x1_next, x2_next])
            else:
                xhat, P = self.update_state(xpred, y, Ppred)
                u_light = self.compute_control(xhat)
                u_zscored = self.zscore(np.concatenate([u_light, np.zeros(self.k_task)]), "u")
                xpred, Ppred = self.predict_state(xhat, u_zscored, P)

            end = time.perf_counter()
            durations.append(end - start)

        durations = np.array(durations[5:])
        median = np.median(durations)
        mad = np.median(np.abs(durations - median))
        # No separate solve time for gradient descent — it's all one computation
        non_solve_time = 0.0
        self._measuring_latency = False
        return median * b2.second, mad / 0.6745 * b2.second, non_solve_time * b2.second
