from dataclasses import dataclass

import brian2.only as b2
import cleo
import numpy as np
from jaxtyping import Float, Int


@dataclass
class TrainedFSNN:
    ng: b2.NeuronGroup
    syn: b2.Synapses


@dataclass
class CleoExperiment:
    """fSNN wrapped with Cleo"""

    fsnn: TrainedFSNN
    light: cleo.light.Light
    probe: cleo.ephys.Probe
    sim: cleo.CLSimulator


@dataclass
class ExptData:
    """should be able to use for sys ID and control experiments data"""

    U: Float[np.ndarray, "trials T k"]
    """Control input in mW/mm2"""
    Y: Float[np.ndarray, "trials T m"]
    """Experimental measurements"""
