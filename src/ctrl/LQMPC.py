import brian2.only as b2
import numpy as np
from attrs import define, field

from src.ctrl import Controller


@define
class LQMPCController(Controller):
    """Linear-quadratic MPC: a convex (CVXPY) QP over the light command, solved on the LINEAR model.

    The NN opsin model is NOT wired into this optimization — LQMPC plans as if light maps linearly to
    state, ignoring the opsin's saturating nonlinearity. On the USID+opsin arm it therefore over-drives
    the light (empirically ~2x the no-opsin BRAID controller), saturates ChR2, and goes negative (Q < 0),
    so it is NOT a relevant controller for USID+opsin research — use NLMPC (or iLQR), which plan on the
    nonlinear opsin (compute_photocurrents). LQMPC remains valid for the no-opsin BRAID baseline.
    """

    def reset(self):
        super().reset()
        self.init_optimization()

    def compute_control(self, xhat):
        if self.t_now < self.t_ctrl_bounds[0] or self.t_now >= self.t_ctrl_bounds[1]:
            return self.zscore(np.zeros(self.k), "u")[: self.k_opto]
        self.update_opt_problem(xhat, self.t_now)
        self.solve_optimization()
        return self.Uopt_opto.value[0]
