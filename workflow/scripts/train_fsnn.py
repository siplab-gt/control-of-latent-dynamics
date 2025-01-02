import os
import sys

import brian2.only as b2
import numpy as np
from cleo.coords import assign_coords_rand_rect_prism

from src.data_classes import fSNN
from src.helpers import seed_everything


def main(condition_vars, params, outfile):
    seed_everything(int(condition_vars["seed"]), "train_fsnn")
    # TODO: calculate volume, num neurons per layer
    thickness = (
        params["layer_boundaries_um"][-1] - params["layer_boundaries_um"][0]
    ) * b2.um
    side_length_um = np.sqrt(float(params["array_side_length_mm"]) * 1000) ** 2 * float(
        condition_vars["network_area_scale"]
    )
    vol = (side_length_um * b2.um) ** 2 * thickness
    n_ng = vol * (params["base_neuron_density"] * b2.mm**-3)
    print(f"side_length_um: {side_length_um}, volume: {vol}, n_ng: {n_ng}")
    ng = b2.NeuronGroup(n_ng, "dv/dt = -v / (10*ms) : 1")
    hor_lim = (-side_length_um / 2, side_length_um / 2)
    assign_coords_rand_rect_prism(
        ng,
        hor_lim,
        hor_lim,
        (
            params["layer_boundaries_um"][0],
            params["layer_boundaries_um"][-1],
        ),
        unit=b2.um,
    )
    syn = b2.Synapses(ng, ng, "w : 1")
    syn.connect(p=1e-4)
    syn.w = 0.5

    net = b2.Network(ng, syn)
    fsnn = fSNN(net, ng, syn)
    fsnn.save(outfile)
    print(f"saved to {outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.output[0])
