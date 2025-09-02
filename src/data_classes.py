from typing import Self
import sys
from dataclasses import dataclass

import brian2.only as b2
import cleo
import dill
import numpy as np
from jaxtyping import Float, Int, UInt


class BaseDataClass:
    def save(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename) -> Self:
        if b2.prefs.codegen.runtime.cython.cache_dir not in sys.path:
            sys.path.append(b2.prefs.codegen.runtime.cython.cache_dir)
        with open(filename, "rb") as f:
            return dill.load(f)


@dataclass
class fSNN(BaseDataClass):
    net: b2.Network
    ng: b2.NeuronGroup
    syn: b2.Synapses
    state_mon: b2.StateMonitor

    def refresh_monitors_for_run(self) -> None:
        """since weakref after unserializing breaks the monitors"""
        state_mon2 = b2.StateMonitor(
            self.ng, self.state_mon.recorded_variables, record=True
        )
        # state_mon2.set_states(self.state_mon.get_states())
        self.net.remove(self.state_mon)
        self.state_mon = state_mon2
        self.net.add(state_mon2)

    def sample_condition(self, rng: np.random.Generator) -> int:
        return rng.choice(8)

    def set_condition(self, condition):
        self.ng.condition = condition


@dataclass
class CleoPrep(BaseDataClass):
    """fSNN wrapped with Cleo"""

    fsnn: fSNN
    light: cleo.light.Light
    probe: cleo.ephys.Probe
    sim: cleo.CLSimulator

    def reset(self):
        """Resets monitors and device histories for fresh run (without changing network state)"""
        self.fsnn.refresh_monitors_for_run()
        self.light.reset()
        self.probe.reset()


@dataclass
class ExptData(BaseDataClass):
    """should be able to use for sys ID and control experiments data"""

    U: Float[np.ndarray, "T k"]
    """Control input in mW/mm2, shape (T, k) array"""
    Y: Float[np.ndarray, "T n"]
    """Experimental measurements, shape (T, n) array"""
    t_ms: Float[np.ndarray, "T"]
    """Time in ms, shape (T,) array"""
    i_trial_boundaries: UInt[np.ndarray, "n_trials 2"]
    """Trial (start, ..., end) indices, shape (n_trials, 2) array"""

    def get_trial_UYt(
        self, trial_idx: int
    ) -> tuple[
        Float[np.ndarray, "T k"], Float[np.ndarray, "T n"], Float[np.ndarray, "T"]
    ]:
        """Get the control input, experimental measurements, and time for a specific trial."""
        start = self.i_trial_starts[trial_idx]
        end = self.i_trial_ends[trial_idx]
        return self.U[start:end], self.Y[start:end], self.t_ms[start:end]
