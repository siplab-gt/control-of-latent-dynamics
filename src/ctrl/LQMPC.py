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
        self.solve_optimization()
        return self.Uopt_opto.value[0]
