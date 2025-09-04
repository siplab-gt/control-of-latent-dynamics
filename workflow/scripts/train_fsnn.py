import os
import sys

import brian2.only as b2
import numpy as np
from cleo.coords import assign_coords_rand_rect_prism

from src.data_classes import fSNN
from src.helpers import seed_everything


def main(cond_params, base_params, outfile):
    seed_everything(int(cond_params["seed"]), "train_fsnn")
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
    b2.prefs.codegen.target = "numpy"
    net.run(1 * b2.ms)
    fsnn = fSNN(net, ng, syn, state_mon)
    print(fsnn)
    fsnn.save(outfile)
    print(f"saved to {outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.output[0])
