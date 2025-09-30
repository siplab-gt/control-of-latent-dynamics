import brian2.only as b2
from attrs import define, field

from src.ctrl import Controller


@define
class OLController(Controller):
    t_prev_soln: b2.Quantity = -18990508 * b2.ms

    def update_with_trial_info(self, t_prev_trial_end, t_trial_start, t_trial_end):
        super().update_with_trial_info(t_prev_trial_end, t_trial_start, t_trial_end)
        # set horizon to cover through end of trial
        self.horizon = int(round((t_trial_end - t_prev_trial_end) / self.sample_period))

    def compute_control(self, xhat):
        if self.i_hzn >= self.horizon:
            self.init_optimization()
            self.update_opt_problem(xhat, self.t_now)
            self.prob.solve()
            self.t_prev_soln = self.t_now
        return self.Uopt_opto.value[self.i_hzn]

    @property
    def i_hzn(self) -> int:
        return int(round((self.t_now - self.t_prev_soln) / self.sample_period))
