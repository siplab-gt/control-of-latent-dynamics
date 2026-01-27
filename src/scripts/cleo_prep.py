import logging
from dataclasses import dataclass, field
from typing import Self

import brian2.only as b2
import cleo

from src.data_classes import BaseDataClass
from src.fsnn import fSNN
from src.helpers import preprocess_params, setup_debugger, setup_logging

log = logging.getLogger(__name__)


@dataclass
class CleoPrep(BaseDataClass):
    """fSNN wrapped with Cleo"""

    fsnn: fSNN
    light: cleo.light.Light
    probe: cleo.ephys.Probe
    sim: cleo.CLSimulator

    mua: cleo.ephys.MultiUnitActivity = None
    lfp: cleo.ephys.LFPSignalBase = None
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
        import logging

        import cleo
        import cleo.opto
        import numpy as np
        from cleo.ephys import (
            MultiUnitActivity,
            Probe,
            RWSLFPSignalFromPSCs,
            TKLFPSignal,
        )

        from src.grouped_light import GroupedLight
        from src.helpers import seed_everything
        from src.interface_array import get_interface_array

        log = logging.getLogger(__name__)

        seed = seed_everything(cond_params.seed, "cleo_prep")
        rng = np.random.default_rng(seed)

        log.debug(f"{cond_params=}")
        light_model = cleo.light.fiber473nm(R0=cond_params.beam_width / 2)
        grid = get_interface_array(
            base_params.layer_boundaries[0] / b2.mm,
            base_params.array_side_length / b2.mm,
            base_params.array_n_per_side,
        )

        # to not have to repeat this step every time we load
        if "light_coords" in cache and "probe_coords" in cache:
            light_coords = cache["light_coords"]
            light_groups = cache["light_groups"]
            probe_coords = cache["probe_coords"]
            print("loaded cached light/probe coords")
        else:
            # separate files so one can change without causing the other to be rerun
            setting = cond_params.setting
            module_name = f"src.cleo_prep_{setting}"
            module = importlib.import_module(module_name)
            light_and_probe_coords = module.light_and_probe_coords

            if cond_params.m == "NA":
                m = 0
            else:
                m = cond_params.m
            light_coords, light_groups, probe_coords = light_and_probe_coords(
                grid, int(cond_params.k), m, rng=rng
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
        sim.inject(light_array, fsnn.ng_main)

        # tune opsin strength relative to input weights
        # making 5 mW/mm2 at 0.5 mm distance equivalent to a typical task input
        typical_light_model = cleo.light.fiber473nm()
        typical_fin_mag = 1
        typical_task_input = typical_fin_mag * np.mean(np.abs(fsnn.u_in))
        typical_distance_from_fiber_mm = 0.5
        typical_transmittance = typical_light_model.transmittance(
            source_coords=[0, 0, 0] * b2.mm,
            source_dir_uvec=np.array([0, 0, 1]),
            target_coords=[0, 0, typical_distance_from_fiber_mm] * b2.mm,
        ).item()
        typical_Irr = (
            base_params.typical_Irr0_mWmm2 * b2.mwatt / b2.mm**2 * typical_transmittance
        )
        I_per_Irr = typical_task_input / typical_Irr
        log.debug(f"{typical_task_input=}, {typical_transmittance=}, {I_per_Irr=}")
        opsin = cleo.opto.ProportionalCurrentOpsin(I_per_Irr=I_per_Irr)
        sim.inject(opsin, fsnn.ng_main)

        mua = None
        lfp = None
        if probe_coords is not None:
            # drop probe coords down into middle of L2/3
            mid_l23_depth = (
                base_params.layer_boundaries[1] - base_params.layer_boundaries[0]
            ) / 2
            probe_coords[:, 2] += mid_l23_depth
            signals = []
            inj_kwargs = {}
            if cond_params.signals in ["spikes", "both"]:
                # use higher cutoff to avoid memory explosion
                mua = MultiUnitActivity(
                    name="mua",
                    recording_recall_cutoff=0.01,
                    r_noise_floor=200 * b2.um,
                    threshold_sigma=3,
                )
                signals.append(mua)
            if cond_params.signals in ["lfp", "both"]:
                # we don't have separate AMPA and GABA currents, but we will
                # use the sum ΣI as a rudimentary LFP proxy (as shown in Mazzoni 2015)
                signals.append(lfp := RWSLFPSignalFromPSCs(name="lfp"))
                # SCLFP works quite a bit better than TKLFP from preliminary experiments.
                # signals.append(lfp := TKLFPSignal(name="lfp"))
                inj_kwargs |= dict(
                    Iampa_var_names=["vinf"],
                    Igaba_var_names=[],
                    tklfp_type="exc",
                    sample_period=fsnn.dt * base_params.subsample_factor,
                )
            probe = Probe(probe_coords, signals=signals)
            sim.inject(probe, fsnn.ng_main, **inj_kwargs)
        else:
            probe = None

        return cls(
            fsnn, light_array, probe, sim, mua, lfp, cond_params, base_params, cache
        )

    def __reduce__(self):
        return (
            self.__class__.build,
            (self.fsnn, self._cond_params, self._base_params, self._cache),
        )


def main(cond_params, base_params, fsnn_file, outfile):
    fsnn = fSNN.load(fsnn_file)
    log.info(f"Loaded fSNN from {fsnn_file}")
    log.info(f"{fsnn=}")
    cleo_expt = CleoPrep.build(fsnn, cond_params, base_params)
    cleo_expt.save(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    log_fh = setup_logging(snakemake.config, snakemake.log[0])
    setup_debugger(snakemake.config["debug"])

    main(
        preprocess_params(snakemake.wildcards),
        preprocess_params(snakemake.params),
        snakemake.input[0],
        snakemake.output[0],
    )
