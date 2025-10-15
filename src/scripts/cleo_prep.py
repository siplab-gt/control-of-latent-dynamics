from dataclasses import dataclass, field
from typing import Self

import brian2.only as b2
import cleo

from src.data_classes import BaseDataClass
from src.log_redirector import redirect_output_to_file
from src.fsnn import fSNN


@dataclass
class CleoPrep(BaseDataClass):
    """fSNN wrapped with Cleo"""

    fsnn: fSNN
    light: cleo.light.Light
    probe: cleo.ephys.Probe
    mua: cleo.ephys.MultiUnitActivity
    sim: cleo.CLSimulator
    _cond_params: dict = None
    _base_params: dict = None
    _cache: dict = field(repr=False, default_factory=dict)

    def reset(self):
        """Resets monitors and device histories for fresh run (without changing network state)"""
        self.light.reset()
        if self.probe is not None:
            self.probe.reset()
        assert self.fsnn.net == self.sim.network

    def run(self, dt, *args, **kwargs):
        self.sim.run(dt, *args, **kwargs)

    @classmethod
    def build(cls, fsnn, cond_params, base_params, cache={}) -> Self:
        import importlib

        import cleo
        import numpy as np

        from src.grouped_light import GroupedLight
        from src.helpers import seed_everything
        from src.interface_array import get_interface_array

        seed = seed_everything(int(cond_params["seed"]), "cleo_prep")
        rng = np.random.default_rng(seed)

        light_model = cleo.light.fiber473nm(
            R0=float(cond_params["beam_width_um"]) * b2.um / 2
        )
        grid = get_interface_array(
            float(base_params["array_side_length_mm"]),
            float(cond_params["total_scale"]),
            int(base_params["array_n_per_side"]),
        )

        # to not have to repeat this step every time we load
        if "light_coords" in cache and "probe_coords" in cache:
            light_coords = cache["light_coords"]
            light_groups = cache["light_groups"]
            probe_coords = cache["probe_coords"]
            print("loaded cached light/probe coords")
        else:
            # separate files so one can change without causing the other to be rerun
            setting = cond_params["setting"]
            module_name = f"src.cleo_prep_{setting}"
            module = importlib.import_module(module_name)
            light_and_probe_coords = module.light_and_probe_coords

            if cond_params["m"] == "NA":
                m = 0
            else:
                m = int(cond_params["m"])
            light_coords, light_groups, probe_coords = light_and_probe_coords(
                grid, int(cond_params["k"]), m, rng=rng
            )
            cache = {
                "light_coords": light_coords,
                "light_groups": light_groups,
                "probe_coords": probe_coords,
            }

        light_array = GroupedLight(
            name="light array",
            coords=light_coords,
            groups=light_groups,
            light_model=light_model,
        )
        sim = cleo.CLSimulator(fsnn.net)
        sim.inject(light_array, fsnn.ng)

        if probe_coords is not None:
            # use higher cutoff to avoid memory explosion
            mua = cleo.ephys.MultiUnitActivity(name="mua", recording_recall_cutoff=0.01)
            probe = cleo.ephys.Probe(probe_coords, signals=[mua])
            sim.inject(probe, fsnn.ng)
        else:
            probe = None
            mua = None

        return cls(fsnn, light_array, probe, mua, sim, cond_params, base_params, cache)

    def __reduce__(self):
        return (
            self.__class__.build,
            (self.fsnn, self._cond_params, self._base_params, self._cache),
        )


def main(cond_params, base_params, fsnn_file, outfile):
    fsnn = fSNN.load(fsnn_file)
    print(f"{fsnn=}")
    cleo_expt = CleoPrep.build(fsnn, cond_params, base_params)
    # print(f"{cleo_expt=}")
    cleo_expt.save(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    with redirect_output_to_file(snakemake.log[0]):
        main(
            snakemake.wildcards,
            snakemake.params,
            snakemake.input[0],
            snakemake.output[0],
        )
