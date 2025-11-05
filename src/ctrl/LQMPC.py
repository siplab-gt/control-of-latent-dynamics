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
        if self.t_now >= self.t_trial_bounds[1]:
            return np.zeros(self.k_opto)
        self.update_opt_problem(xhat, self.t_now)
        self.prob.solve(warm_start=True)
        return self.Uopt_opto.value[0]
