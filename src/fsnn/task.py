from dataclasses import dataclass
from abc import ABC, abstractmethod


import numpy as np
import brian2.only as b2


@dataclass
class fSNNTask:
    dt: b2.Quantity
    n_in: int
    n_out: int
    g: float  # feedback gain in spiking network
    muf: float  # recurrent inhibition
    gs: float  # gain of slow random synapses
    gf: float  # gain of fast random synpases
    name: str

    @abstractmethod
    def generate_trial(self, rng, base_params, task_params):
        """returns fin,fout,T_trial,cond,i_trial_start_end,nsteps_iti"""

    @abstractmethod
    def postprocess_trial_info(
        self,
        fin,
        fout,
        T_trial,
        cond,
        i_trial_start_end,
        nsteps_iti,
    ):
        """returns ytilde, f, fin"""
