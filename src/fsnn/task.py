from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import brian2.only as b2
import numpy as np


@dataclass
class fSNNTask:
    u: np.ndarray = field(repr=False)
    """(n_ng, n_factors), maps from factors to SNN neurons"""
    u_in: np.ndarray = field(repr=False)
    """(n_ng, n_in), maps from inputs to SNN"""
    dt: b2.Quantity

    @property
    def n_in(self) -> int: ...
    @property
    def n_out(self) -> int: ...
    @property
    def n_factors(self) -> int: ...

    g: float  # feedback gain in spiking network
    muf: float  # recurrent inhibition
    gs: float  # gain of slow random synapses
    gf: float  # gain of fast random synpases

    @property
    def name(self) -> str:
        return type(self).__name__.replace("Task", "").lower()

    @abstractmethod
    def generate_trial(self, rng):
        """returns ytilde, fout, fin, T_trial, cond, i_trial_start_end, nsteps_iti"""
