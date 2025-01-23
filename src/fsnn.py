from dataclasses import dataclass
from typing import Self

import brian2.only as b2
import numpy as np
from cleo.coords import assign_coords_rand_rect_prism

from src.helpers import seed_everything


@dataclass
class fSNN:
    net: b2.Network
    ng: b2.NeuronGroup
    syn: b2.Synapses
    state_mon: b2.StateMonitor

    def _b2_store_name(self, pkl_filename):
        assert pkl_filename[-4:] == ".pkl"
        return pkl_filename[:-4] + ".b2store"

    def save(self, filename):
        print(f"{self.__getstate__()=}")
        with open(filename, "wb") as f:
            dill.dump(self, f)
        self.net.store(filename=self._b2_store_name(filename))

    @classmethod
    def load(cls, filename) -> Self:
        with open(filename, "rb") as f:
            fsnn = dill.load(f)
        print(f"{fsnn.net=}")
        fsnn.net.restore(filename=fsnn._b2_store_name(filename))
        print(f"{fsnn.net=}")
        return fsnn


def prep_fsnn(cond_params, base_params) -> fSNN:
    seed_everything(int(cond_params["seed"]), "prep_fsnn")
    # TODO: calculate volume, num neurons per layer
    thickness = (
        base_params["layer_boundaries_um"][-1] - base_params["layer_boundaries_um"][0]
    ) * b2.um
    side_length_um = np.sqrt(
        float(base_params["array_side_length_mm"]) * 1000
    ) ** 2 * float(cond_params["network_area_scale"])
    vol = (side_length_um * b2.um) ** 2 * thickness
    n_ng = vol * (base_params["base_neuron_density"] * b2.mm**-3)
    print(f"side_length_um: {side_length_um}, volume: {vol}, n_ng: {n_ng}")
    ng = b2.NeuronGroup(
        n_ng, "dv/dt = -v / (10*ms) : 1 \n dY/dt = -(Y + xi*sqrt(3*ms)) / (20*ms) : 1"
    )
    hor_lim = (-side_length_um / 2, side_length_um / 2)
    assign_coords_rand_rect_prism(
        ng,
        hor_lim,
        hor_lim,
        (
            base_params["layer_boundaries_um"][0],
            base_params["layer_boundaries_um"][-1],
        ),
        unit=b2.um,
    )
    syn = b2.Synapses(ng, ng, "w : 1")
    syn.connect(p=1e-4)
    syn.w = 0.5

    state_mon = b2.StateMonitor(ng, "Y", record=True)

    net = b2.Network(ng, syn, state_mon)
    fsnn = fSNN(net, ng, syn, state_mon)
    return fsnn
