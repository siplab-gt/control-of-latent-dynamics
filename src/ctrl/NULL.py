import numpy as np

from src.ctrl import Controller


class NULLController(Controller):
    def compute_control(self, xhat):
        return self.zscore(np.zeros(self.k), "u")[: self.k_opto]
