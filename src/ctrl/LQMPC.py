import brian2.only as b2
import numpy as np
from attrs import define, field

from src.ctrl import Controller


@define
class LQMPCController(Controller):
    def reset(self):
        super().reset()
        self.init_optimization()

    def compute_control(self, xhat):
        if self.t_now < self.t_ctrl_bounds[0] or self.t_now >= self.t_ctrl_bounds[1]:
            return self.zscore(np.zeros(self.k), "u")[: self.k_opto]
        self.update_opt_problem(xhat, self.t_now)
        self.prob.solve(
            warm_start=True,
            enforce_dpp=True,
            # lower infeasibility threshold (from default 1e-4) since we know the problem is feasible
            # this goes against the convention that smaller tolerances are more sensitive
            eps_prim_inf=1e-10,
            eps_dual_inf=1e-10,
        )
        assert self.prob.status not in ["infeasible", "unbounded"], (
            f"{self.prob.solver_stats=}"
        )
        return self.Uopt_opto.value[0]
