import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cleo
import numpy as np
from brian2 import Quantity, mm2, ms, mwatt, second


@dataclass
@ABC
class Controller(cleo.ioproc.LatencyIOProcessor):
    k: int
    m: int
    rng: np.random.Generator
    u_max_mWmm2: float
    baseline_latency: Quantity
    compute_latency: Quantity = None
    spike_counts: list = field(default_factory=list)

    def measure_and_set_latency(self):
        durations = []
        xpred = np.zeros(self.m)
        u = self.rng.uniform(0, self.u_max_mWmm2, self.k) * mwatt / mm2
        for i in range(100):
            avg_rate = 20 / second
            y = self.rng.poisson(lam=avg_rate * self.sample_period)

            start = time.perf_counter()
            xhat = self.update_state(xpred, y)
            u = self.compute_control(xhat)
            xpred = self.predict_state(xhat, u)
            end = time.perf_counter()
            durations.append(end - start)

        self.compute_latency = np.mean(durations) * second

    @abstractmethod
    def update_state(self, xpred, y):
        # Placeholder: simple identity filter
        return xpred

    @abstractmethod
    def predict_state(self, xhat, u):
        # Placeholder: simple identity prediction
        return xhat

    @abstractmethod
    def compute_control(self, y_hat):
        pass

    def process(self, state_dict, t_samp):
        t, i, counts = state_dict["Probe"]["mua"]
        self.spike_counts.append(counts)
        value = self.rng.uniform(0, self.u_max_mWmm2, self.k) * mwatt / mm2
        return {"light array": value}, t_samp
