from attrs import define, field
import brian2.only as b2

from src.ctrl import Controller


@define
class OLController(Controller):
    def update_state(self, xpred, y):
        return xpred

    def predict_state(self, xhat, u):
        return xhat

    def compute_control(self, xhat):
        # Placeholder: random control
        u = self.rng.uniform(0, self.u_max_mWmm2, self.k_opto)
        return u
