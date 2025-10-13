import importlib
import os
import sys
from dataclasses import dataclass, field

import brian2.only as b2
import brian2cuda  # noqa: F401
import numpy as np
from jaxtyping import UInt
from matplotlib import pyplot as plt

from src.data_classes import BaseDataClass
from src.log_redirector import redirect_output_to_file


@dataclass
class fSNN(BaseDataClass):
    net: b2.Network
    ng: b2.NeuronGroup
    syn: b2.Synapses
    state_mon: b2.StateMonitor
    v_mon: b2.StateMonitor
    n_pcs: int
    i_trial_boundaries: UInt[np.ndarray, "n_trials n_events"]
    """Trial (start, ..., end) indices, shape (n_trials, n_events) array"""
    _mode: str = "train"
    _save_data: bool = True
    _cond_params: dict = None
    _base_params: dict = None
    _state_vars_to_save: list = field(
        default_factory=lambda: {"ng": ["v", "X"], "syn": ["w"]}
    )

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
        return out

    @property
    def u_task_(self):
        """callable with signature t -> task-related input"""
        return lambda t: self.u_task(t)

    def sample_condition(self, rng: np.random.Generator) -> int:
        return rng.choice(8)

    def set_condition(self, condition):
        self.ng.condition = condition

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @property
    def condition(self) -> int:
        return self.ng.condition

    @condition.setter
    def condition(self, value: int):
        self.ng.condition = value

    @property
    def monitors(self):
        return [self.state_mon, self.v_mon]

    def start_recording(self):
        self._save_data = True
        for mon in self.monitors:
            mon.active = True

    def disable_recording(self):
        self._save_data = False
        for mon in self.monitors:
            mon.active = False

    def run_trial(self, rng=None, condition=None):
        if condition is None:
            assert rng is not None
            condition = self.sample_condition(rng)
        self.condition = condition
        iti_min, iti_max = self._base_params["t_intertrial_ms"]
        t_intertrial = (
            rng.uniform(
                self._base_params["t_intertrial_min_ms"],
                self._base_params["t_intertrial_max_ms"],
            )
            * b2.ms
        )

    def __reduce__(self):
        net_state = {}
        for attr, state_vars in self._state_vars_to_save.items():
            net_obj = self.__dict__[attr]
            net_state[attr] = {var: getattr(net_obj, var) for var in state_vars}

        return (self.__class__.build, (self._cond_params, self._base_params, net_state))

    @classmethod
    def build(cls, cond_params, base_params, net_state={}) -> "fSNN":
        # lame, but need to import inside here for constructor to work when unpickling
        import numpy as np
        from cleo.coords import assign_coords_rand_rect_prism, coords_from_ng

        from src.helpers import seed_everything

        b2.defaultclock.dt = float(base_params.dt_ms) * b2.ms

        seed = seed_everything(int(cond_params["seed"]), "train_fsnn")
        rng = np.random.default_rng(seed)

        # TODO: calculate volume, num neurons per layer
        thickness = (
            base_params["layer_boundaries_um"][-1]
            - base_params["layer_boundaries_um"][0]
        ) * b2.um
        side_length_um = np.sqrt(
            float(base_params["array_side_length_mm"]) * 1000
        ) ** 2 * float(cond_params["network_area_scale"])
        vol = (side_length_um * b2.um) ** 2 * thickness
        n_ng = vol * (base_params["base_neuron_density"] * b2.mm**-3)
        n_ng = int(round(n_ng))
        print(f"side_length_um: {side_length_um}, volume: {vol}, n_ng: {n_ng}")
        hor_lim = (-side_length_um / 2, side_length_um / 2)

        ############# start fSNN code ##############

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
        print(f"{ng.N=}")
        syn = b2.Synapses(ng, ng, "w : 1", name="synapses")
        n_syn = rng.poisson(lam=1e-4 * n_ng * n_ng)
        print(f"{n_syn=}")
        i = rng.integers(0, n_ng, size=n_syn)
        j = rng.integers(0, n_ng, size=n_syn)
        syn.connect(i=i, j=j)
        # can't do this, will be different depending on device
        # syn.connect(p=1e-4)
        syn.w = 0.5

        samp_period = float(base_params["sample_period_ms"]) * b2.ms
        state_mon = b2.StateMonitor(
            ng, ["X", "Z"], record=np.arange(10), dt=samp_period
        )
        v_mon = b2.StateMonitor(ng, "v", record=np.arange(10))

        net = b2.Network(ng, syn, state_mon, v_mon)

        syn.w = "randn()"

        fsnn = cls(
            net,
            ng,
            syn,
            state_mon,
            v_mon,
            int(cond_params["n_pcs"]),
            cond_params,
            base_params,
        )

        for attr, state_vars in net_state.items():
            net_obj = getattr(fsnn, attr)
            print(f"loading state for {attr}: {list(state_vars.keys())}")
            for var, values in state_vars.items():
                getattr(net_obj, var)[:] = values

        print(f"{coords_from_ng(ng)[:10]=}")
        return fsnn


def main(cond_params, base_params, train_device, fsnn_outfile):
    outdir = os.path.dirname(fsnn_outfile)
    if train_device == "cuda":
        b2.set_device("cuda_standalone", build_on_run=False)
    else:
        assert train_device == "cython"
        b2.prefs.codegen.target = "cython"

    # TODO: will have to change for multi-task supposrt
    factors, emg = importlib.import_module("src.fsnn").load_data(cond_params.task)

    fsnn = fSNN.build(cond_params, base_params, factors, emg)

    fsnn.disable_recording()
    fsnn.mode = "demean"
    for i in range(int(base_params["n_trials_demean"])):
        fsnn.run_trial()
    fsnn.mode = "train"
    for i in range(int(base_params["n_trials_train"])):
        fsnn.run_trial()
    fsnn.mode = "test"
    fsnn.start_recording()
    for i in range(int(base_params["n_trials_test"])):
        fsnn.run_trial()

    if train_device == "cuda":
        print("compiling cuda code")
        b2.device.build(
            directory=f"{outdir}/brian2cuda_output",
            results_directory=outdir,
            run=False,
        )
        print("done building")

        b2.device.run()

    print("done running")
    print(f"{fsnn.ng.X[0:10]=}")
    plt.plot(fsnn.state_mon.t / b2.ms, fsnn.state_mon.X.T)
    plt.savefig(f"{outdir}/example_X.png")
    plt.close()
    plt.plot(fsnn.v_mon.t / b2.ms, fsnn.v_mon.v.T / b2.mV)
    plt.savefig(f"{outdir}/example_v.png")
    fsnn.save(fsnn_outfile)
    print(f"saved to {fsnn_outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    with redirect_output_to_file(snakemake.log[0]):
        main(
            snakemake.wildcards,
            snakemake.params,
            snakemake.config["train_device"],
            snakemake.output[0],
        )
