import importlib
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
    mua: cleo.ephys.MultiUnitActivity
    sim: cleo.CLSimulator

    def reset(self):
        """Resets monitors and device histories for fresh run (without changing network state)"""
        self.fsnn.refresh_monitors_for_run()
        self.refresh_monitors_for_run()
        self.light.reset()
        self.probe.reset()
        assert self.fsnn.net == self.sim.network

    def refresh_monitors_for_run(self) -> None:
        """since weakref after unserializing breaks the monitors"""
        # spkmon in MUA
        assert len(self.mua._monitors) == 1
        spkmon = self.mua._monitors[0]
        new_spkmon = b2.SpikeMonitor(spkmon.source)
        self.fsnn.net.remove(spkmon)
        self.mua._monitors[0] = new_spkmon
        self.fsnn.net.add(new_spkmon)

    def run(self, dt, *args, **kwargs):
        self.sim.run(dt, *args, **kwargs)


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
    setting = cond_params["setting"]
    module_name = f"src.cleo_prep_{setting}"
    module = importlib.import_module(module_name)
    light_and_probe_coords = module.light_and_probe_coords

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

    if probe_coords is not None:
        mua = cleo.ephys.MultiUnitActivity(name="mua")
        probe = cleo.ephys.Probe(probe_coords, signals=[mua])
        sim.inject(probe, fsnn.ng)
    else:
        probe = None

    cleo_expt = CleoPrep(fsnn, light_array, probe, mua, sim)
    cleo_expt.save(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.input[0], snakemake.output[0])
