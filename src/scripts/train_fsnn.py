import os
import sys
from dataclasses import dataclass

import brian2.only as b2
import brian2cuda  # noqa: F401
import numpy as np
from cleo.coords import assign_coords_rand_rect_prism

from src.data_classes import BaseDataClass
from src.helpers import seed_everything


@dataclass
class fSNN(BaseDataClass):
    net: b2.Network
    ng: b2.NeuronGroup
    syn: b2.Synapses
    state_mon: b2.StateMonitor

    def refresh_monitors_for_run(self) -> None:
        """since weakref after unserializing breaks the monitors"""
        new_clock = b2.Clock(dt=self.state_mon.clock.dt)
        new_clock._set_t_update_dt(self.state_mon.clock.variables["t"].get_value())
        state_mon2 = b2.StateMonitor(
            self.ng, self.state_mon.recorded_variables, record=True, clock=new_clock
        )
        # state_mon2.set_states(self.state_mon.get_states())
        self.net.remove(self.state_mon)
        self.state_mon = state_mon2
        self.net.add(state_mon2)

    def sample_condition(self, rng: np.random.Generator) -> int:
        return rng.choice(8)

    def set_condition(self, condition):
        self.ng.condition = condition


def main(cond_params, base_params, fsnn_outfile):
    seed = seed_everything(int(cond_params["seed"]), "train_fsnn")
    rng = np.random.default_rng(seed)

    outdir = os.path.dirname(fsnn_outfile)
    # b2.prefs.codegen.target = "cython"
    b2.set_device("cuda_standalone", build_on_run=False)

    # TODO: calculate volume, num neurons per layer
    thickness = (
        base_params["layer_boundaries_um"][-1] - base_params["layer_boundaries_um"][0]
    ) * b2.um
    side_length_um = np.sqrt(
        float(base_params["array_side_length_mm"]) * 1000
    ) ** 2 * float(cond_params["network_area_scale"])
    vol = (side_length_um * b2.um) ** 2 * thickness
    n_ng = vol * (base_params["base_neuron_density"] * b2.mm**-3)
    n_ng = int(round(n_ng))
    print(f"side_length_um: {side_length_um}, volume: {vol}, n_ng: {n_ng}")
    ng = b2.NeuronGroup(
        n_ng,
        """dv/dt = -(v + 70*mvolt + (Iopto_exc + Iopto_inh)*500*Mohm) / (10*ms) : volt
        dY/dt = -(Y + xi*sqrt(3*ms)) / (20*ms) : 1
        Iopto_exc : amp
        Iopto_inh : amp
        condition : 1
        """,
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

    print(f"{ng.N=}")
    state_mon = b2.StateMonitor(ng, "Y", record=np.arange(n_ng), dt=1 * b2.ms)

    net = b2.Network(ng, syn, state_mon)
    net.run(1 * b2.ms)

    b2.device.build(
        directory=f"{outdir}/brian2cuda_output",
        results_directory=outdir,
        run=False,
    )
    print("done building")

    b2.device.run()
    syn.w = rng.normal(0, 1, size=len(syn))
    print("done running")
    fsnn = fSNN(net, ng, syn, state_mon)
    print(fsnn)
    fsnn.save(fsnn_outfile)
    print(f"saved to {fsnn_outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.output[0])
