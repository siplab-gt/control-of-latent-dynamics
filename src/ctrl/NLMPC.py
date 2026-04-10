"""
Nonlinear MPC controller using PyTorch gradient descent.

Optimizes light inputs directly through the opsin model + linear state
dynamics. Replaces the CVXPY QP with a differentiable forward model:

    light → opsin GRU → photocurrents → B' @ p → state update

Box constraints [0, u_max] are enforced by clamping after each gradient step.
Warm-starts from the previous solution (shifted by one step) for fast
convergence (typically 5-10 iterations).
"""
import logging

import brian2.only as b2
import numpy as np
import torch
from attrs import define, field
from brian2 import mm2, ms, mwatt

from src.ctrl import Controller

log = logging.getLogger(__name__)


def _rollout_state(x0, photocurrents, A11, A22, K1, K2, C1, C2, A12,
                   B1_prime, B2_prime, ref_trajectory, ref_type, Cz=None):
    """Roll out state dynamics through the MPC horizon.

    Args:
        x0: (n,) initial state [x1; x2] (torch tensor)
        photocurrents: (H, k_opto) photocurrents from opsin (torch tensor)
        A11..B2_prime: state-space matrices (torch tensors)
        ref_trajectory: (H, n_r) reference in z-scored space (torch tensor)
        ref_type: "X" or "Z"
        Cz: (n_r, n) behavioral decoder (torch tensor), needed if ref_type="Z"

    Returns:
        tracking_cost: scalar (torch tensor)
    """
    n1 = A11.shape[0]
    H = photocurrents.shape[0]

    x1 = x0[:n1]
    x2 = x0[n1:]
    cost = torch.tensor(0.0, dtype=x0.dtype, device=x0.device)

    for t in range(H):
        p_t = photocurrents[t]

        # State update (observer form, using y = C1@x1 + C2@x2 as predicted y)
        y_pred = C1 @ x1 + C2 @ x2

        x1_next = A11 @ x1 + K1 @ y_pred + B1_prime @ p_t
        if A12 is not None:
            x1_next = x1_next + A12 @ x2
        innovation = y_pred - C1 @ x1
        x2_next = A22 @ x2 + K2 @ innovation + B2_prime @ p_t

        x1, x2 = x1_next, x2_next

        # Tracking cost
        x_full = torch.cat([x1, x2])
        if ref_type == "Z" and Cz is not None:
            ctrl_var = Cz @ x_full
        else:
            ctrl_var = x_full
        cost = cost + torch.sum((ctrl_var - ref_trajectory[t]) ** 2)

    return cost / H


@define
class NLMPCController(Controller):
    """Nonlinear MPC using PyTorch gradient descent through opsin + dynamics."""
    nlmpc_iters: int = 10
    nlmpc_lr: float = 0.5

    # Internal state (not constructor args)
    _u_light_warm: np.ndarray = field(default=None, init=False, repr=False)
    _torch_matrices: dict = field(factory=dict, init=False, repr=False)
    _opsin_model: object = field(default=None, init=False, repr=False)

    def reset(self):
        super().reset()
        self._u_light_warm = None
        self._setup_torch_matrices()

    def _setup_torch_matrices(self):
        """Extract state-space matrices from LDS and convert to torch tensors."""
        lds = self.lds
        if not hasattr(lds, 'A11'):
            return

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

        if hasattr(lds, 'Cz') and lds.Cz is not None:
            self._torch_matrices['Cz'] = torch.tensor(lds.Cz, dtype=torch.float32)
        elif hasattr(lds, 'Cz1') and lds.Cz1 is not None:
            Cz = np.hstack([lds.Cz1, lds.Cz2])
            self._torch_matrices['Cz'] = torch.tensor(Cz, dtype=torch.float32)

        if hasattr(lds, 'opsin_model') and lds.opsin_model is not None:
            self._opsin_model = lds.opsin_model
            self._opsin_model.eval()
            for p in self._opsin_model.parameters():
                p.requires_grad_(False)

    def _run_opsin_forward(self, u_light):
        """Run opsin model: (H, k_opto) light → (H, k_opto) photocurrents.

        Processes each channel independently through the GRU, matching
        USIDModel.compute_photocurrents() structure.
        """
        H, k_opto = u_light.shape
        # Reshape to (k_opto, H, 1) for per-channel GRU processing
        u_flat = u_light.T.unsqueeze(-1)  # (k_opto, H, 1)
        p_flat = self._opsin_model(u_flat)  # (k_opto, H)
        return p_flat.T  # (H, k_opto)

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

        u_light = torch.tensor(warm, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([u_light], lr=self.nlmpc_lr)

        x0 = torch.tensor(xhat, dtype=torch.float32)

        # Reference trajectory over horizon
        t_horizon = self.t_now + np.arange(1, H + 1) * self.sample_period
        ref_np = self.reference_zscored(t_horizon)
        ref = torch.tensor(ref_np, dtype=torch.float32)

        mat = self._torch_matrices

        for i in range(self.nlmpc_iters):
            optimizer.zero_grad()

            # Forward: light → opsin → photocurrents → state rollout
            photocurrents = self._run_opsin_forward(u_light)

            tracking_cost = _rollout_state(
                x0, photocurrents,
                mat['A11'], mat['A22'], mat['K1'], mat['K2'],
                mat['C1'], mat['C2'], mat['A12'],
                mat['B1_prime'], mat['B2_prime'],
                ref, self.ref_type, mat.get('Cz'),
            )

            # Control effort cost (penalizes light power)
            effort_cost = torch.sum(u_light ** 2) / (H * k_opto)

            loss = self.Qcost * tracking_cost / self.n_r + self.Rcost * effort_cost
            loss.backward()
            optimizer.step()

            # Box constraints
            with torch.no_grad():
                u_light.clamp_(0.0, self.u_max_mWmm2)

        # Store for warm-starting next step
        self._u_light_warm = u_light.detach().numpy().copy()

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

        # Observer update: run opsin on applied light to get photocurrent for B' @ p
        if self.state_estimation == "observer":
            n1 = self.lds.n1
            y_zscored = self.zscore(y, "y")
            with torch.no_grad():
                light_tensor = torch.tensor(
                    u_light.reshape(1, 1, -1), dtype=torch.float32
                )
                p_tensor = self._run_opsin_forward(light_tensor.squeeze(0))
                p_current = p_tensor.squeeze().numpy()
            x1_next, x2_next = self._observer_step(
                self.lds, self.xpred[:n1], self.xpred[n1:], y_zscored, p_current,
            )
            self.xpred = np.concatenate([x1_next, x2_next])
        elif self.state_estimation == "kalman":
            # For Kalman, we'd need to handle the nonlinear input mapping.
            # For now, fall back to raw light z-scored (approximate).
            u_zscored = self.zscore(
                np.concatenate([u_light, self.u_task(t_samp)]), "u"
            )
            self.xpred, self.Ppred = self.predict_state(xhat, u_zscored, P)

        return {"light array": u_light * mwatt / mm2}, t_samp + self.sample_latency()

    def measure_latency(self):
        """Measure computation latency for the nonlinear MPC."""
        self._measuring_latency = True
        durations = []
        xpred = np.zeros(self.n)

        for i in range(100):
            avg_rate = 20 / b2.second
            y = self.latency_rng.poisson(lam=avg_rate * self.sample_period)

            import time
            start = time.perf_counter()

            if self.state_estimation == "observer":
                xhat = xpred.copy()
                u_light = self.compute_control(xhat)
                n1 = self.lds.n1
                y_zscored = self.zscore(y, "y")
                with torch.no_grad():
                    light_t = torch.tensor(
                        u_light.reshape(1, 1, -1), dtype=torch.float32
                    )
                    p_t = self._run_opsin_forward(light_t.squeeze(0))
                    p_np = p_t.squeeze().numpy()
                x1_next, x2_next = self._observer_step(
                    self.lds, xpred[:n1], xpred[n1:], y_zscored, p_np,
                )
                xpred = np.concatenate([x1_next, x2_next])
            else:
                xhat, P = self.update_state(xpred, y, self.Ppred)
                u_light = self.compute_control(xhat)
                u_zscored = self.zscore(np.concatenate([u_light, np.zeros(self.k_task)]), "u")
                xpred, _ = self.predict_state(xhat, u_zscored, P)

            end = time.perf_counter()
            durations.append(end - start)

        durations = np.array(durations[5:])
        median = np.median(durations)
        mad = np.median(np.abs(durations - median))
        # No separate solve time for gradient descent — it's all one computation
        non_solve_time = 0.0
        self._measuring_latency = False
        return median * b2.second, mad / 0.6745 * b2.second, non_solve_time * b2.second
