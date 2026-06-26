"""
iLQR (iterative Linear-Quadratic Regulator) MPC controller.

Solves the same trajectory-optimization problem as NLMPC, but replaces the
flat Adam gradient descent with a structured iLQR backward+forward sweep:
quadratic value-function expansion, Gauss-Newton-style Q_uu inversion with
Levenberg-Marquardt regularization, and a backtracking line search over the
(k, K) feedback policy. Convergence is typically 3-10 iterations vs.
Adam's 20+.

Three rollout modes mirror NLMPC exactly:

  1. **opsin_subspace** — USID with `use_opsin=True`. The NN opsin's GRU
     hidden state is held fixed PER ITERATION at the values from that
     iteration's forward pass (standard "frozen-h" approximation for
     RNN-in-the-loop iLQR). Each new iteration's forward pass updates the
     hidden state, so the linearization re-aligns with the new trajectory.
  2. **linearized_subspace** — USID with `use_opsin=False`. Dynamics are
     linear in z-scored light; F_u is constant (= B' / u_std).
  3. **linearized_unified** — BRAID. Linear in z-scored light; F_u = B_opto
     / u_std.

Box constraint u ∈ [0, u_max] is enforced by clipping inside the forward-
pass line search (after applying the feedback policy). The regularization
μ on Q_uu is adaptive: shrinks on accepted steps, grows on rejected ones.

Author note on the design choice (frozen-h vs. augmented state): augmenting
the iLQR state with the GRU hidden (~2·k·32 dims = 2048 for k=32) is correct
in principle but blows up backward-pass matrix ops to O((n+n_h)^3) per step.
The frozen-h approximation captures the GRU's cross-time coupling through
the iteration loop's forward re-simulation instead of through the per-step
linearization, and is standard in practical iLQR-with-RNN literature.
"""
import logging
import time

import brian2.only as b2
import numpy as np
import torch
from attrs import define, field
from brian2 import mm2, ms, mwatt

from src.ctrl import Controller
from src.ctrl._cz_helpers import apply_cz, build_cz_state, cz_jacobian, has_cz

log = logging.getLogger(__name__)


def _build_A_full_subspace(A11, A22, K1, K2, C1, C2, A12):
    """Build the full observer-form A for backward-pass linearization.

    The subspace observer recurrence used in `_rollout_state` (NLMPC) is:
        y_pred = C1·x1 + C2·x2
        x1[t+1] = A11·x1 + K1·y_pred + B1'·u (+ A12·x2)
        x2[t+1] = A22·x2 + K2·(y_pred - C1·x1) + B2'·u
                = A22·x2 + K2·C2·x2 + B2'·u

    Expanding y_pred:
        x1[t+1] = (A11 + K1·C1)·x1 + (K1·C2 + A12)·x2 + B1'·u
        x2[t+1] = (A22 + K2·C2)·x2 + B2'·u

    So A_full =
        [[A11 + K1·C1,   K1·C2 + A12 (or K1·C2 if A12 is None)],
         [0,             A22 + K2·C2                          ]]

    Note A12 may be None — handled inline.
    """
    n1 = A11.shape[0]
    n2 = A22.shape[0]
    A_full = torch.zeros(n1 + n2, n1 + n2, dtype=A11.dtype, device=A11.device)
    A_full[:n1, :n1] = A11 + K1 @ C1
    upper_right = K1 @ C2
    if A12 is not None:
        upper_right = upper_right + A12
    A_full[:n1, n1:] = upper_right
    A_full[n1:, n1:] = A22 + K2 @ C2
    return A_full


def _step_subspace(x, p, A11, A22, K1, K2, C1, C2, A12, B1_prime, B2_prime):
    """Single subspace observer-form step matching NLMPC's `_rollout_state`.

    Returns x_next of shape (n,). Differentiable through p (and through A
    matrices, though we only need d/du in practice).
    """
    n1 = A11.shape[0]
    x1, x2 = x[:n1], x[n1:]
    y_pred = C1 @ x1 + C2 @ x2
    x1_next = A11 @ x1 + K1 @ y_pred + B1_prime @ p
    if A12 is not None:
        x1_next = x1_next + A12 @ x2
    innovation = y_pred - C1 @ x1  # = C2·x2
    x2_next = A22 @ x2 + K2 @ innovation + B2_prime @ p
    return torch.cat([x1_next, x2_next])


@define
class ILQRController(Controller):
    """iLQR MPC with mode parity to NLMPC.

    Hyperparameters (all configurable from `ctrl_expts_iLQR` block):

      ilqr_max_iters: max iLQR iterations per MPC step (default 10).
      ilqr_tol: relative cost tolerance for convergence (default 1e-4).
      ilqr_mu_init: initial Q_uu regularization (default 1e-3).
      ilqr_mu_min: lower bound on μ (default 1e-6).
      ilqr_mu_max: upper bound on μ; abort iter if exceeded (default 1e6).
      ilqr_mu_factor: μ multiplier on reject / divisor on accept (default 10
        for grow, 2 for shrink — both controlled by this single knob via
        grow=factor, shrink=sqrt(factor) heuristic; see _adapt_mu).
      ilqr_alpha_schedule: line-search backtracking factors (default
        [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]).
    """
    ilqr_max_iters: int = 10
    ilqr_tol: float = 1e-4
    ilqr_mu_init: float = 1e-3
    ilqr_mu_min: float = 1e-6
    ilqr_mu_max: float = 1e6
    ilqr_mu_factor: float = 10.0
    ilqr_alpha_schedule: tuple = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)

    _u_light_warm: np.ndarray = field(default=None, init=False, repr=False)
    _torch_matrices: dict = field(factory=dict, init=False, repr=False)
    _opsin_model: object = field(default=None, init=False, repr=False)
    _opsin_hidden: object = field(default=None, init=False, repr=False)
    _subspace_mode: bool = field(default=False, init=False, repr=False)
    _has_opsin: bool = field(default=False, init=False, repr=False)
    _cz_state: dict = field(factory=dict, init=False, repr=False)

    def sample_latency(self) -> b2.Quantity:
        """No CVXPY solver — sample purely from baseline + compute mean/std."""
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
        self._setup_torch_matrices()

    def _setup_torch_matrices(self):
        """Mode detection + torch tensor setup. Identical to NLMPC's helper."""
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
            # Precompute the constant Fx = A_full used by the backward pass.
            mat = self._torch_matrices
            self._torch_matrices['A_full'] = _build_A_full_subspace(
                mat['A11'], mat['A22'], mat['K1'], mat['K2'],
                mat['C1'], mat['C2'], mat['A12'],
            )
            self._torch_matrices['B_full_prime'] = torch.cat(
                [mat['B1_prime'], mat['B2_prime']], dim=0
            )  # (n, k_opto)
        else:
            self._torch_matrices = {
                'A': torch.tensor(lds.A, dtype=torch.float32),
                'B_opto': torch.tensor(lds.B_opto, dtype=torch.float32),
                'C': torch.tensor(lds.C, dtype=torch.float32),
            }
        # Single source of truth for Cz dispatch (linear matrix vs nonlinear
        # torch module). See src/ctrl/_cz_helpers.py.
        self._cz_state = build_cz_state(lds)

        if self._has_opsin:
            self._opsin_model = lds.opsin_model
            self._opsin_model.eval()
            for p in self._opsin_model.parameters():
                p.requires_grad_(False)
            # Diagonal-Jacobian assumption: NN opsin must process each light
            # channel as an independent GRU batch element (input_size=1).
            # Crashing loud here beats silently using a wrong Fu if a
            # multi-channel-input opsin is ever attached.
            assert self._opsin_model.gru.input_size == 1, (
                f"iLQR requires per-channel-independent NN opsin "
                f"(GRU input_size=1), got input_size="
                f"{self._opsin_model.gru.input_size}. The diagonal-Jacobian "
                f"optimization assumes cross-channel grads are 0."
            )
        else:
            self._torch_matrices['u_mean_opto'] = torch.tensor(
                lds.u_mean[: self.k_opto], dtype=torch.float32
            )
            self._torch_matrices['u_std_opto'] = torch.tensor(
                lds.u_std[: self.k_opto], dtype=torch.float32
            )

    def _run_opsin_forward(self, u_light, h0=None):
        """Run opsin model: (H, k_opto) light → (H, k_opto) photocurrents.

        Mirrors NLMPC. Stateful forward returns updated hidden state for
        propagation through the MPC window.
        """
        u_flat = u_light.T.unsqueeze(-1)  # (k_opto, H, 1)
        p_flat, h_out = self._opsin_model.forward_with_hidden(u_flat, h0)
        return p_flat.T, h_out

    def _forward_rollout(self, x0, u_light, h0):
        """Forward-simulate trajectory under control sequence u_light.

        Returns:
            x_traj: (H+1, n) state trajectory (no grad)
            p_traj: (H, k_opto) photocurrent trajectory (no grad), or None
                in linearized modes
            h_traj: list of length (H+1) of GRU hidden states (or None),
                each a tensor copy detached from graph
            cost: scalar tracking + effort cost (no grad)
            ref: (H, n_r) reference trajectory in z-scored space (no grad)
        """
        H = u_light.shape[0]
        mat = self._torch_matrices

        # Reference trajectory over horizon
        t_horizon = self.t_now + np.arange(1, H + 1) * self.sample_period
        ref_np = self.reference_zscored(t_horizon)
        ref = torch.tensor(ref_np, dtype=torch.float32)

        with torch.no_grad():
            x_traj = [x0.clone()]
            p_traj = None
            h_traj = None

            if self._has_opsin:
                # Per-step propagation so we can capture h at each timestep
                # (needed for the frozen-h Jacobian computation in the backward
                # setup).
                p_traj = torch.zeros(H, self.k_opto, dtype=torch.float32)
                h_traj = [h0.detach() if h0 is not None else None]
                h = h0
                for t in range(H):
                    p_t, h = self._run_opsin_forward(
                        u_light[t:t+1], h0=h
                    )
                    p_traj[t] = p_t[0]
                    h_traj.append(h.detach() if h is not None else None)
                    x_next = _step_subspace(
                        x_traj[-1], p_traj[t],
                        mat['A11'], mat['A22'], mat['K1'], mat['K2'],
                        mat['C1'], mat['C2'], mat['A12'],
                        mat['B1_prime'], mat['B2_prime'],
                    )
                    x_traj.append(x_next)
            elif self._subspace_mode:
                u_zscored = (u_light - mat['u_mean_opto']) / mat['u_std_opto']
                for t in range(H):
                    x_next = _step_subspace(
                        x_traj[-1], u_zscored[t],
                        mat['A11'], mat['A22'], mat['K1'], mat['K2'],
                        mat['C1'], mat['C2'], mat['A12'],
                        mat['B1_prime'], mat['B2_prime'],
                    )
                    x_traj.append(x_next)
            else:
                u_zscored = (u_light - mat['u_mean_opto']) / mat['u_std_opto']
                A = mat['A']
                B = mat['B_opto']
                for t in range(H):
                    x_next = A @ x_traj[-1] + B @ u_zscored[t]
                    x_traj.append(x_next)

            x_traj = torch.stack(x_traj)  # (H+1, n)

        cost = self._total_cost(x_traj, u_light, ref)
        return x_traj, p_traj, h_traj, cost, ref

    def _total_cost(self, x_traj, u_light, ref):
        """Sum cost over horizon: Q · tracking + R · effort, both normalized.

        Matches NLMPC's loss exactly: see `NLMPC._rollout_state`. Cz dispatch
        goes through `_cz_helpers.apply_cz` so linear and nonlinear (torch
        module) Cz are handled uniformly.
        """
        H = u_light.shape[0]
        if self.ref_type == "Z" and has_cz(self._cz_state):
            ctrl_var = apply_cz(self._cz_state, x_traj[1:])
        else:
            ctrl_var = x_traj[1:]
        tracking = torch.sum((ctrl_var - ref) ** 2) / H
        effort = torch.sum(u_light ** 2) / (H * self.k_opto)
        return (self.Qcost * tracking / self.n_r) + (self.Rcost * effort)

    def _compute_Fu_per_step(self, u_light, h_traj):
        """Compute F_u = ∂(x[t+1])/∂(u[t]) per step.

        Three regimes:
          • opsin_subspace: F_u_t = B_full' · (∂p[t]/∂u[t] | h_traj[t]).
            The Jacobian ∂p/∂u is computed via autograd at each step with
            h fixed to h_traj[t] (frozen-h linearization).
          • linearized_subspace: F_u = B_full' / u_std (constant).
          • linearized_unified: F_u = B_opto / u_std (constant).

        Returns: list of length H of (n, k_opto) tensors.
        """
        H = u_light.shape[0]
        mat = self._torch_matrices

        if not self._has_opsin:
            # Linear in z-scored light. F_u = B / u_std (same per step).
            if self._subspace_mode:
                B = mat['B_full_prime']
            else:
                B = mat['B_opto']
            Fu_const = B / mat['u_std_opto'].unsqueeze(0)  # (n, k_opto)
            return [Fu_const for _ in range(H)]

        # opsin_subspace: per-step Jacobian. The NN opsin processes each
        # channel independently (k_opto is the GRU's batch dim, see
        # `_run_opsin_forward` and OpsinClusterNN's invariant), so
        # ∂p_i/∂u_j = 0 for i ≠ j. The Jacobian is diagonal — we recover
        # just the diagonal `g_i = ∂p_i/∂u_i` via a single backward pass
        # with `grad_outputs=ones`. Cross-channel grads would sum to the
        # same result since they're all zero. Then F_u = B' · diag(g)
        # = B' * g (broadcasting along last axis). This is O(H) backward
        # passes per iLQR iteration instead of O(H · k_opto).
        Fu_list = []
        B_full = mat['B_full_prime']  # (n, k_opto)
        for t in range(H):
            h_prev = h_traj[t]
            u_in = u_light[t].detach().clone().requires_grad_(True)
            p_t, _ = self._run_opsin_forward(u_in.unsqueeze(0), h0=h_prev)
            p_out = p_t[0]  # (k_opto,)
            grad_diag = torch.autograd.grad(
                p_out, u_in,
                grad_outputs=torch.ones_like(p_out),
                create_graph=False, retain_graph=False,
            )[0]
            Fu_list.append(B_full * grad_diag.unsqueeze(0))  # (n, k_opto)
        return Fu_list

    def _backward_pass(self, x_traj, u_light, ref, Fu_list, mu):
        """Standard iLQR backward DP with Levenberg-Marquardt regularization.

        Cost approximation per step (using x[t+1]'s contribution to tracking):
            l(x_{t+1}, u_t) = Q/n_r · ||Cz·x_{t+1} - r_t||^2 + R/k · ||u_t||^2

        Linearized dynamics: x_{t+1} = Fx · x_t + Fu_t · u_t + const.
            Fx = A_full (constant for subspace) or A (unified) — both regimes
            already torch tensors in _torch_matrices.

        For LQR convenience we shift the cost to be in terms of (x_t, u_t)
        instead of (x_{t+1}, u_t): l(x_t, u_t) → l(x_{t+1}, u_t) where the
        x dependence enters via x_{t+1} = Fx·x_t + Fu·u_t. This is the
        "running cost shifted by one step" convention common in iLQR papers.

        Returns: (K_list, k_list, expected_dV1, expected_dV2)
            K_list: list of (k_opto, n) feedback gains
            k_list: list of (k_opto,) feedforward updates
            expected_dV1, expected_dV2: scalars for expected cost change
                (used to compute line-search acceptance ratio).
        """
        H = u_light.shape[0]
        mat = self._torch_matrices
        Fx = mat['A_full'] if self._subspace_mode else mat['A']
        n = Fx.shape[0]
        k_opto = u_light.shape[1]

        # Tracking-cost Hessian Lxx depends on Cz dispatch:
        #   • linear Cz   → Lxx is constant per step (Cz.T @ Cz).
        #   • nonlinear Cz → Gauss-Newton approximation J(x).T @ J(x) at
        #     each x_{t+1}; computed per step inside the backward loop.
        # Build a static `Lxx_const` for the linear paths; for nonlinear it
        # is set to None and recomputed inside the loop.
        cz_is_nonlinear = self._cz_state.get('is_nonlinear', False)
        if not cz_is_nonlinear and has_cz(self._cz_state):
            Cz = cz_jacobian(self._cz_state, x_traj[1])  # constant matrix
            Lxx_const = (2.0 * self.Qcost / self.n_r) * (Cz.T @ Cz)
        elif not has_cz(self._cz_state):
            # ref_type='X' path: identity Lxx (Cz is the identity map).
            Lxx_const = (2.0 * self.Qcost / self.n_r) * torch.eye(n, dtype=Fx.dtype)
        else:
            Lxx_const = None  # filled per step in nonlinear case
        Luu = (2.0 * self.Rcost / self.k_opto) * torch.eye(
            k_opto, dtype=Fx.dtype
        )

        # Backward pass starts from V[H] = 0 (no terminal cost).
        Vx = torch.zeros(n, dtype=Fx.dtype)
        Vxx = torch.zeros(n, n, dtype=Fx.dtype)

        K_list = [None] * H
        k_list = [None] * H
        dV1 = 0.0  # sum of k^T · Q_u
        dV2 = 0.0  # 0.5 * sum of k^T · Q_uu · k

        for t in reversed(range(H)):
            Fu = Fu_list[t]

            # Stage cost gradients evaluated at (x_{t+1}, u_t) = (x_traj[t+1], u_light[t]):
            x_next = x_traj[t + 1]
            r_t = ref[t]
            if cz_is_nonlinear:
                # Nonlinear Cz: Jacobian and residual at this state.
                J_cz = cz_jacobian(self._cz_state, x_next)        # (m_z, n)
                resid = apply_cz(self._cz_state, x_next) - r_t    # (m_z,)
                lx_next = (2.0 * self.Qcost / self.n_r) * (J_cz.T @ resid)
                # Gauss-Newton Lxx at this step (drops 2nd-order ∂²Cz/∂x²).
                Lxx = (2.0 * self.Qcost / self.n_r) * (J_cz.T @ J_cz)
            elif has_cz(self._cz_state):
                Cz_mat = self._cz_state['Cz']
                lx_next = (2.0 * self.Qcost / self.n_r) * (Cz_mat.T @ (Cz_mat @ x_next - r_t))
                Lxx = Lxx_const
            else:
                lx_next = (2.0 * self.Qcost / self.n_r) * (x_next - r_t)
                Lxx = Lxx_const
            lu = (2.0 * self.Rcost / self.k_opto) * u_light[t]

            # Pull derivatives back from x_{t+1} through dynamics to (x_t, u_t).
            # ∂l/∂x_t = Fx^T · lx_next; ∂l/∂u_t = lu + Fu^T · lx_next.
            l_xt = Fx.T @ lx_next
            l_ut = lu + Fu.T @ lx_next
            # Hessians: ∂²l/∂x_t² = Fx^T · Lxx · Fx; ∂²l/∂u_t² = Luu + Fu^T·Lxx·Fu;
            #           ∂²l/∂u_t∂x_t = Fu^T · Lxx · Fx.
            l_xxt = Fx.T @ Lxx @ Fx
            l_uut = Luu + Fu.T @ Lxx @ Fu
            l_uxt = Fu.T @ Lxx @ Fx

            # Q-function expansion
            Qx = l_xt + Fx.T @ Vx
            Qu = l_ut + Fu.T @ Vx
            Qxx = l_xxt + Fx.T @ Vxx @ Fx
            Quu = l_uut + Fu.T @ Vxx @ Fu
            Qux = l_uxt + Fu.T @ Vxx @ Fx

            # Levenberg-Marquardt: add μ·I to Quu
            Quu_reg = Quu + mu * torch.eye(k_opto, dtype=Fx.dtype)

            # Symmetrize for numerical stability (cheap insurance)
            Quu_reg = 0.5 * (Quu_reg + Quu_reg.T)
            Qxx = 0.5 * (Qxx + Qxx.T)

            # Cholesky-solve for feedback policy. Fall back to LU if Cholesky
            # fails (Quu not PD even after regularization — rare).
            try:
                L = torch.linalg.cholesky(Quu_reg)
                k_t = -torch.cholesky_solve(Qu.unsqueeze(-1), L).squeeze(-1)
                K_t = -torch.cholesky_solve(Qux, L)
            except Exception:
                # Caller will increase mu and retry; signal failure with None.
                return None, None, None, None

            K_list[t] = K_t
            k_list[t] = k_t

            # Value function update (standard form):
            #   V_x  = Qx + K^T·Quu·k + K^T·Qu + Qux^T·k
            #   V_xx = Qxx + K^T·Quu·K + K^T·Qux + Qux^T·K
            Vx = Qx + K_t.T @ Quu @ k_t + K_t.T @ Qu + Qux.T @ k_t
            Vxx = Qxx + K_t.T @ Quu @ K_t + K_t.T @ Qux + Qux.T @ K_t
            Vxx = 0.5 * (Vxx + Vxx.T)

            # Expected change in cost (for line-search acceptance ratio)
            dV1 = dV1 + (k_t @ Qu).item()
            dV2 = dV2 + 0.5 * (k_t @ Quu @ k_t).item()

        return K_list, k_list, dV1, dV2

    def _forward_with_policy(self, x0, u_old, x_old, K_list, k_list, alpha, h0):
        """Apply feedback policy with step alpha; re-simulate forward.

        Box constraint u ∈ [0, u_max] enforced by clipping after each step.

        Returns: (u_new, x_new, cost_new) where x_new is (H+1, n).
        """
        H = u_old.shape[0]
        mat = self._torch_matrices

        with torch.no_grad():
            u_new = torch.zeros_like(u_old)
            x_new = torch.zeros_like(x_old)
            x_new[0] = x0
            h = h0
            for t in range(H):
                dx = x_new[t] - x_old[t]
                u_t = u_old[t] + alpha * k_list[t] + K_list[t] @ dx
                u_t = torch.clamp(u_t, 0.0, self.u_max_mWmm2)
                u_new[t] = u_t

                if self._has_opsin:
                    p_t, h = self._run_opsin_forward(u_t.unsqueeze(0), h0=h)
                    x_next = _step_subspace(
                        x_new[t], p_t[0],
                        mat['A11'], mat['A22'], mat['K1'], mat['K2'],
                        mat['C1'], mat['C2'], mat['A12'],
                        mat['B1_prime'], mat['B2_prime'],
                    )
                elif self._subspace_mode:
                    u_z = (u_t - mat['u_mean_opto']) / mat['u_std_opto']
                    x_next = _step_subspace(
                        x_new[t], u_z,
                        mat['A11'], mat['A22'], mat['K1'], mat['K2'],
                        mat['C1'], mat['C2'], mat['A12'],
                        mat['B1_prime'], mat['B2_prime'],
                    )
                else:
                    u_z = (u_t - mat['u_mean_opto']) / mat['u_std_opto']
                    x_next = mat['A'] @ x_new[t] + mat['B_opto'] @ u_z
                x_new[t + 1] = x_next

        # Reference is consistent across line-search trials within one MPC step.
        # t_horizon recomputed locally for clarity; cost is small
        # (np.arange + one dict lookup).
        t_horizon = self.t_now + np.arange(1, H + 1) * self.sample_period
        ref = torch.tensor(self.reference_zscored(t_horizon), dtype=torch.float32)
        cost_new = self._total_cost(x_new, u_new, ref)
        return u_new, x_new, cost_new

    def compute_control(self, xhat):
        if self.t_now < self.t_ctrl_bounds[0] or self.t_now >= self.t_ctrl_bounds[1]:
            return np.zeros(self.k_opto)

        H = self.n_horizon
        k_opto = self.k_opto

        # Warm-start from previous solution
        if self._u_light_warm is not None and self._u_light_warm.shape[0] == H:
            warm = np.roll(self._u_light_warm, -1, axis=0)
            warm[-1] = warm[-2] if H > 1 else 0.0
        else:
            warm = np.zeros((H, k_opto))

        u_light = torch.tensor(warm, dtype=torch.float32)
        x0 = torch.tensor(xhat, dtype=torch.float32)

        # h0 for planning is the real opsin state (detached so iLQR doesn't
        # backprop through tissue history). None in non-opsin modes.
        h0_planning = (
            self._opsin_hidden.detach() if self._opsin_hidden is not None else None
        )

        # Initial forward pass
        x_traj, p_traj, h_traj, cost, ref = self._forward_rollout(
            x0, u_light, h0_planning
        )

        mu = self.ilqr_mu_init

        for it in range(self.ilqr_max_iters):
            # Per-step Fu (frozen-h for opsin mode)
            Fu_list = self._compute_Fu_per_step(u_light, h_traj)

            # Backward pass with current μ; grow μ and retry on Cholesky failure
            K_list = k_list = None
            backward_attempts = 0
            while K_list is None and mu <= self.ilqr_mu_max and backward_attempts < 6:
                K_list, k_list, dV1, dV2 = self._backward_pass(
                    x_traj, u_light, ref, Fu_list, mu
                )
                if K_list is None:
                    mu = mu * self.ilqr_mu_factor
                    backward_attempts += 1
            if K_list is None:
                log.warning(
                    f"iLQR: backward pass failed at iter {it} (mu={mu:.2e}); "
                    f"returning current best."
                )
                break

            # Forward line search. We accept the first alpha that monotonically
            # decreases the cost (the standard expected-reduction ratio test
            # from Tassa et al. is bypassed because box-clipping invalidates
            # the unconstrained-Newton expected-reduction model).
            accepted = False
            rel_improve = 0.0
            for alpha in self.ilqr_alpha_schedule:
                u_new, x_new, cost_new = self._forward_with_policy(
                    x0, u_light, x_traj, K_list, k_list, alpha, h0_planning
                )
                if cost_new.item() < cost.item():
                    cost_prev = cost.item()
                    u_light = u_new
                    # Re-roll under accepted u_new to capture per-step h for
                    # next iteration's frozen-h Jacobians. _forward_with_policy
                    # carries h forward but doesn't store per-step values.
                    x_traj, p_traj, h_traj, cost, ref = self._forward_rollout(
                        x0, u_light, h0_planning
                    )
                    accepted = True
                    rel_improve = (cost_prev - cost.item()) / max(abs(cost_prev), 1e-9)
                    mu = max(self.ilqr_mu_min, mu / self.ilqr_mu_factor)
                    break

            if accepted and rel_improve < self.ilqr_tol:
                break

            if not accepted:
                # All alphas failed → increase μ and try again next iter.
                mu = min(self.ilqr_mu_max, mu * self.ilqr_mu_factor)
                if mu >= self.ilqr_mu_max:
                    log.debug(f"iLQR: μ saturated at {mu:.2e} on iter {it}; halting.")
                    break

        self._u_light_warm = u_light.detach().numpy().copy()
        return self._u_light_warm[0]

    def process(self, state_dict, t_samp):
        """Mirror NLMPC.process exactly — trial boundary handling, observer
        update, opsin GRU advance. Raw light returned to the simulator."""
        t_samp = round(t_samp / ms) * ms
        n_trials = len(self.cleo_prep.fsnn.i_trial_boundaries)
        if t_samp >= self.t_ctrl_bounds[1]:
            if self.i_current_trial < n_trials - 1:
                log.info(f"finished control trial {self.i_current_trial}")
                self.i_current_trial += 1
                assert t_samp < self.t_ctrl_bounds[1], (
                    f"{t_samp=} not less than {self.t_ctrl_bounds[1]=} after incrementing trial"
                )
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

        if self.state_estimation == "kalman":
            xhat, P = self.update_state(self.xpred, y, self.Ppred)
        elif self.state_estimation == "observer":
            xhat = self.xpred.copy()
        elif self.state_estimation == "ground truth":
            xhat = self.zscore(self.cleo_prep.fsnn.ng_y.y[:], "x")

        self._Xhat.append(xhat)
        self._R.append(self.reference_zscored(t_samp))

        u_light = self.compute_control(xhat)
        u_light = np.clip(u_light, 0, self.u_max_mWmm2)

        if self.state_estimation == "observer" and self._has_opsin:
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
            u_opto_zscored = (
                u_light - self.lds.u_mean[: self.k_opto]
            ) / self.lds.u_std[: self.k_opto]
            u = np.concatenate([u_opto_zscored, self.u_task_zscored(t_samp)])
            self.xpred, self.Ppred = self.predict_state(xhat, u, P)

        return {"light array": u_light * mwatt / mm2}, t_samp + self.sample_latency()

    def measure_latency(self):
        """Mode-aware latency measurement mirroring NLMPC's pattern."""
        self._measuring_latency = True
        durations = []
        xpred = np.zeros(self.n)
        opsin_hidden = None
        Ppred = np.eye(self.n) if not self._subspace_mode else None

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
        non_solve_time = 0.0
        self._measuring_latency = False
        return median * b2.second, mad / 0.6745 * b2.second, non_solve_time * b2.second
