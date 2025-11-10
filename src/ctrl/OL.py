import logging

import brian2.only as b2
import numpy as np
from attrs import define, field

from src.ctrl import Controller

log = logging.getLogger(__name__)


@define
class OLController(Controller):
    t_prev_soln: b2.Quantity = -18990508 * b2.ms

    @property
    def n_horizon(self) -> int:
        return int(
            round((self.t_ctrl_bounds[1] - self.t_ctrl_bounds[0]) / self.sample_period)
        )

    def compute_control(self, xhat):
        if self.t_now < self.t_ctrl_bounds[0] or self.t_now >= self.t_ctrl_bounds[1]:
            return self.zscore(np.zeros(self.k), "u")[: self.k_opto]
        if self.i_hzn == 0:
            self.init_optimization()
            self.update_opt_problem(xhat, self.t_now)
            self.prob.solve(
                solver="osqp", max_iter=int(1e6), ignore_dpp=True, verbose=True
            )
            assert self.prob.status not in ["infeasible", "unbounded"]
            self.t_prev_soln = self.t_now
        return self.Uopt_opto.value[self.i_hzn]

    @property
    def i_hzn(self) -> int:
        return int(round((self.t_now - self.t_ctrl_bounds[0]) / self.sample_period))
