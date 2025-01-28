from typing import Self
import sys
from dataclasses import dataclass

import brian2.only as b2
import cleo
import dill
import numpy as np
from jaxtyping import Float, Int


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


@dataclass
class CleoPrep(BaseDataClass):
    """fSNN wrapped with Cleo"""

    fsnn: fSNN
    light: cleo.light.Light
    probe: cleo.ephys.Probe
    sim: cleo.CLSimulator


@dataclass
class ExptData(BaseDataClass):
    """should be able to use for sys ID and control experiments data"""

    U: Float[np.ndarray, "trials T k"]
    """Control input in mW/mm2"""
    Y: Float[np.ndarray, "trials T m"]
    """Experimental measurements"""
