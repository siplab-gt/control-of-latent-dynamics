from dataclasses import dataclass

import brian2.only as b2
import cleo
import numpy as np

from src.data_classes import BaseDataClass
from src.grouped_light import GroupedLight
from src.interface_array import get_interface_array
from src.scripts.train_fsnn import fSNN


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


def main(cond_params, base_params, fsnn_file, outfile):
    fsnn = fSNN.load(fsnn_file)
    light_model = cleo.light.fiber473nm(
        R0=float(cond_params["beam_width_um"]) * b2.um / 2
    )
    grid = get_interface_array(
        float(base_params["array_side_length_mm"]),
        float(cond_params["total_scale"]),
        int(base_params["array_n_per_side"]),
    )

    # separate files so one can change without causing the other to be rerun
    if cond_params["setting"] == "sufficient_stim":
        from src.cleo_prep_sufficient_stim import light_and_probe_coords
    elif cond_params["setting"] == "optimal_stim_rec":
        from src.cleo_prep_optimal_stim_rec import light_and_probe_coords

    light_coords, light_groups, probe_coords = light_and_probe_coords(
        grid, int(cond_params["k"])
    )

    array = GroupedLight(
        name="light array",
        coords=light_coords,
        groups=light_groups,
        light_model=light_model,
    )
    sim = cleo.CLSimulator(fsnn.net)
    sim.inject(array, fsnn.ng)

    if probe_coords is not None:
        probe = cleo.ephys.Probe(probe_coords)
        sim.inject(probe, fsnn.ng)
    else:
        probe = None

    print(probe)
    cleo_expt = CleoPrep(fsnn, array, probe, sim)
    cleo_expt.save(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.input[0], snakemake.output[0])
