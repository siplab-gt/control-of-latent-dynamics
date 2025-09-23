import os
import sys
from dataclasses import dataclass

import brian2.only as b2
import brian2cuda  # noqa: F401
import numpy as np
from cleo.coords import assign_coords_rand_rect_prism
from matplotlib import pyplot as plt

from src.data_classes import BaseDataClass
from src.helpers import seed_everything


@dataclass
class fSNN(BaseDataClass):
    net: b2.Network
    ng: b2.NeuronGroup
    syn: b2.Synapses
    state_mon: b2.StateMonitor
    v_mon: b2.StateMonitor
    n_pcs: int

    @property
    def k_task(self) -> int:
        return 3  # TODO: get from network

    def u_task(self, t: b2.Quantity) -> np.ndarray:
        """task-related inputs, shape ([T], n_task_inputs) array"""
        # TODO: get from network
        out = np.array(
            [
                20 * np.sin(2 * np.pi * t / (100 * b2.ms)),
                20 * np.sin(2 * np.pi * t / (200 * b2.ms) + np.pi / 4),
                20 * np.sin(2 * np.pi * t / (300 * b2.ms) + np.pi / 2),
            ]
        ).T
        T = len(t) if hasattr(t, "__len__") else 1
        assert out.shape == (T, self.k_task), f"{out.shape=} {T=} {self.k_task=}"
        return out

    @property
    def u_task_(self):
        """callable with signature t -> task-related input"""
        return lambda t: self.u_task(t)

    def refresh_monitors_for_run(self) -> None:
        """since weakref after unserializing breaks the monitors"""
        new_clock = b2.Clock(dt=self.state_mon.clock.dt)
        new_clock._set_t_update_dt(self.state_mon.clock.variables["t"].get_value())

        for state_mon_name in ["state_mon", "v_mon"]:
            state_mon = getattr(self, state_mon_name)
            state_mon2 = b2.StateMonitor(
                self.ng, state_mon.recorded_variables, record=True, clock=new_clock
            )
            # state_mon2.set_states(state_mon.get_states())
            self.net.remove(state_mon)
            setattr(self, state_mon_name, state_mon2)
            self.net.add(state_mon2)

    def sample_condition(self, rng: np.random.Generator) -> int:
        return rng.choice(8)

    def set_condition(self, condition):
        self.ng.condition = condition


def main(cond_params, base_params, train_device, fsnn_outfile):
    seed = seed_everything(int(cond_params["seed"]), "train_fsnn")
    rng = np.random.default_rng(seed)

    outdir = os.path.dirname(fsnn_outfile)
    if train_device == "cuda":
        b2.set_device("cuda_standalone", build_on_run=False)
    else:
        assert train_device == "cython"
        b2.prefs.codegen.target = "cython"

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
        """dv/dt = (-(v + 70*mvolt) + (Iopto_exc + Iopto_inh)*500*Mohm) / (10*ms) : volt
        dX/dt = (-X + i + 10*xi*sqrt(10*ms) + (v/mvolt + 70)) / (20*ms) : 1
        # dY/dt = 1 / ms : 1
        # Y = i + t / ms : 1
        Iopto_exc : amp
        Iopto_inh : amp
        Z = cos(X) : 1  # "behavior"
        condition : 1
        """,
        threshold="v > -50*mvolt",
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
    samp_period = float(base_params["sample_period_ms"]) * b2.ms
    state_mon = b2.StateMonitor(ng, ["X", "Z"], record=np.arange(10), dt=samp_period)
    v_mon = b2.StateMonitor(ng, "v", record=np.arange(10))

    net = b2.Network(ng, syn, state_mon, v_mon)
    net.run(10 * b2.ms)

    if train_device == "cuda":
        b2.device.build(
            directory=f"{outdir}/brian2cuda_output",
            results_directory=outdir,
            run=False,
        )
        print("done building")

        b2.device.run()

    syn.w = rng.normal(0, 1, size=len(syn))
    print("done running")
    print(f"{ng.X[0:10]=}")
    plt.plot(state_mon.t / b2.ms, state_mon.X.T)
    plt.savefig(f"{outdir}/example_X.png")
    plt.close()
    plt.plot(v_mon.t / b2.ms, v_mon.v.T / b2.mV)
    plt.savefig(f"{outdir}/example_v.png")
    fsnn = fSNN(net, ng, syn, state_mon, v_mon, int(cond_params["n_pcs"]))
    print(fsnn)
    fsnn.save(fsnn_outfile)
    print(f"saved to {fsnn_outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(
        snakemake.wildcards,
        snakemake.params,
        snakemake.config["train_device"],
        snakemake.output[0],
    )
