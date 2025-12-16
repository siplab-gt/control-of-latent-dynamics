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
        return self.t_to_i_ctrl(self.t_ctrl_bounds[1] - self.t_ctrl_bounds[0])

    def compute_control(self, xhat):
        if self.i_hzn < 0 or self.i_hzn >= self.n_horizon:
            return self.zscore(np.zeros(self.k), "u")[: self.k_opto]
        if self.i_hzn == 0:
            self.init_optimization()
            self.update_opt_problem(xhat, self.t_now)
            self.solve_optimization("OL")
            self.t_prev_soln = self.t_now
        assert self.i_hzn < self.n_horizon, f"{self.i_hzn=} >= {self.n_horizon=}"
        return self.Uopt_opto.value[self.i_hzn]

    @property
    def i_hzn(self) -> int:
        return self.t_to_i_ctrl(self.t_now - self.t_ctrl_bounds[0])
