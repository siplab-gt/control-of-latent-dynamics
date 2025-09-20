import time
from abc import ABC, abstractmethod
from collections.abc import Callable

import cleo
import numpy as np
from attrs import define, field
from BRAID.BRAIDModel import BRAIDModel
from brian2 import Quantity, mm2, ms, mwatt, second


@define
class Controller(cleo.ioproc.LatencyIOProcessor):
    k_opto: int = None
    """number of optogenetic inputs"""
    k_task: int = None
    """number of task-related inputs"""
    m: int = None
    """number of measured outputs (e.g. spike channels)"""
    rng: np.random.Generator = None
    u_max_mWmm2: float = None
    baseline_latency: Quantity = None
    braid_model: BRAIDModel = None
    compute_latency: Quantity = None
    spike_counts: list = field(factory=list)
    xpred: np.ndarray = None
    reference: Callable[[Quantity], Quantity] = field(default=lambda t: 0)
    u_task: Callable[[Quantity], np.ndarray] = field(default=lambda t: np.zeros(0))
    """callable with signature t -> task inputs (shape ([T], k_task) array)"""

    # don't know why I need to make this a field to avoid attrs error
    _needs_off_schedule_sample: bool = False

    @property
    def k(self):
        return self.k_opto + self.k_task

    def measure_latency(self):
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

        return np.mean(durations) * second

    @abstractmethod
    def update_state(self, xpred, y):
        # Placeholder: simple identity filter
        pass

    @abstractmethod
    def predict_state(self, xhat, u):
        # Placeholder: simple identity prediction
        pass

    @abstractmethod
    def compute_control(self, y_hat):
        pass

    def process(self, state_dict, t_samp):
        t, i, counts = state_dict["Probe"]["mua"]
        self.spike_counts.append(counts)
        xhat = self.update_state(self.xpred, counts)
        uopto = self.compute_control(xhat)
        uopto = np.clip(uopto, 0, self.u_max_mWmm2)
        u = np.concatenate((uopto, self.u_task(t_samp)))
        self.xpred = self.predict_state(xhat, u)
        return {"light array": uopto * mwatt / mm2}, t_samp + self.latency

    @property
    def latency(self):
        return self.baseline_latency + self.compute_latency
