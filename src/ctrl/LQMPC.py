import brian2.only as b2
import numpy as np
from attrs import define, field

from src.ctrl import Controller


@define
class LQMPCController(Controller):
    def compute_control(self, xhat):
        self.update_opt_problem(xhat, self.sim.network.t)
        self.prob.solve(warm_start=True)
        return self.Uopt_opto.values[0]
