from dataclasses import dataclass

import brian2.only as b2
import cleo
import numpy as np

from src.fsnn import fSNN
from src.grouped_light import GroupedLight
from src.helpers import seed_everything
from src.interface_array import get_interface_array


@dataclass
class CleoPrep:
    """fSNN wrapped with Cleo"""

    fsnn: fSNN
    light_array: cleo.light.Light
    elec_array: cleo.ephys.Probe
    sim: cleo.CLSimulator


def cleo_prep(fsnn: fSNN, cond_params: dict, config: dict) -> CleoPrep:
    seed_everything(int(cond_params["seed"]), "cleo_prep")
    light_model = cleo.light.fiber473nm(
        R0=float(cond_params["beam_width_um"]) * b2.um / 2
    )
    grid = get_interface_array(
        float(config["array_side_length_mm"]),
        float(cond_params["total_scale"]),
        int(config["array_n_per_side"]),
    )

    # separate files so one can change without causing the other to be rerun
    if cond_params["setting"] == "sufficient_stim":
        from src.cleo_prep_sufficient_stim import light_and_probe_coords
    elif cond_params["setting"] == "optimal-stim-rec":
        from src.cleo_prep_optimal_stim_rec import light_and_probe_coords

    light_coords, light_groups, probe_coords = light_and_probe_coords(
        grid, int(cond_params["k"])
    )

    light_array = GroupedLight(
        name="light array",
        coords=light_coords,
        groups=light_groups,
        light_model=light_model,
    )
    sim = cleo.CLSimulator(fsnn.net)
    sim.inject(light_array, fsnn.ng)

    if probe_coords:
        probe = cleo.ephys.Probe(probe_coords, name="elec array")
        sim.inject(probe, fsnn.ng)
    else:
        probe = None

    return CleoPrep(fsnn, light_array, probe, sim)
