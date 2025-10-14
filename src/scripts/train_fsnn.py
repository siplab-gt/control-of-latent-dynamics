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
from src.helpers import numerize
from src.log_redirector import redirect_output_to_file


def nsteps_large_timed_arrays(base_params, task_params):
    largest_n_trials = np.max(
        [
            base_params["train_fsnn"][f"n_trials_{phase}"]
            for phase in ["demean", "train", "test"]
        ]
    )
    largest_n_steps = (
        (
            task_params["iti_min_s"]
            + task_params["iti_lambda_s"] * 1.5
            + np.sum(task_params["t_event_s"])
        )
        * b2.second
        / (base_params["dt_ms"] * b2.ms)
    )
    largest_n_steps = int(np.round(largest_n_steps))
    return largest_n_trials * largest_n_steps


@dataclass
class fSNN(BaseDataClass):
    net: b2.Network
    ng: b2.NeuronGroup
    syn: b2.Synapses
    state_mon: b2.StateMonitor
    v_mon: b2.StateMonitor
    n_pcs: int
    i_trial_boundaries: UInt[np.ndarray, "n_trials n_events"] = field(
        default_factory=list
    )
    """Trial (start, ..., end) indices, shape (n_trials, n_events) array"""
    conditions: list = field(default_factory=list)
    emg: dict = field(default_factory=dict)
    factors: dict = field(default_factory=dict)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
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
    def set_condition(self, value: int):
        self.ng.condition = value

    @property
    def monitors(self):
        return [self.state_mon, self.v_mon]

    def start_recording(self):
        self._save_data = True
        for mon in self.monitors:
            mon.active = True
        self.conditions = []
        self.i_trial_boundaries = []

    def disable_recording(self):
        self._save_data = False
        for mon in self.monitors:
            mon.active = False

    @property
    def dt(self) -> b2.Quantity:
        dt = self._base_params["dt_ms"] * b2.ms
        assert dt == b2.defaultclock.dt
        return dt

    @property
    def T_trial(self) -> b2.Quantity:
        return (
            np.round(np.sum(self._base_params["t_event_s"]) * b2.second / self.dt)
            * self.dt
        )

    def generate_trial(self):
        """returns fin,fout,TTrial,cond,i_trial_start_end,ITIb,ITI"""
        if True:  # single-task
            task = self._cond_params["task"]
        else:
            raise NotImplementedError("multi-task not implemented yet")
        fn = importlib.import_module(task).generate_trial
        return fn(
            self.rng,
            self._base_params,
            self._base_params["train_fsnn"][task],
            self.emg[task],
        )

    def run_trial(self, trial_params=None):
        if trial_params is None:
            fin, fout, T_trial, condition, i_trial_bounds, iti_b, iti = (
                self.generate_trial()
            )
        else:
            fin, fout, T_trial, condition, i_trial_bounds, iti_b, iti = trial_params
        kwargs = dict(report_period=self._base_params["report_period_s"] * b2.second)
        self.fin.values = fin
        self.fout.values = fout
        # TODO: ytilde TimedArray?
        # need this so TimedArrays can work with respect to trial start
        self.ng.t_trial_start = self.net.t
        self.net.run(T_trial, **kwargs)

        if self._save_data:
            self.conditions.append(condition)
            self.i_trial_boundaries.append(i_trial_bounds)

    def run_trials(self, n_trials, mode, record=False, continue_state=False):
        """needed for CUDA standalone training: mainly to prep TimedArrays
        with all trials in advance"""
        # needed to make changes to variables from run to run
        run_args = self.set_mode(mode)
        run_args |= self.set_record(record)

        fin_all, fout_all, T_trial_all, condition, i_trial_bounds, iti_b, iti = (
            self.generate_trial()
        )
        for i in range(1, n_trials):
            nsteps_prev = len(fin_all)
            fin, fout, T_trial, condition, i_trial_bounds, iti_b, iti = (
                self.generate_trial()
            )
            fin_all = np.vstack([fin_all, fin])
            fout_all = np.vstack([fout_all, fout])
            T_trial_all += T_trial
            if self._save_data:
                self.conditions.append(condition)
                i_start = i_trial_bounds[0] + nsteps_prev
                i_end = i_trial_bounds[1] + nsteps_prev
                self.i_trial_boundaries.append((i_start, i_end))

        self.fin.values = fin_all
        self.fout.values = fout_all
        self.ng.t_trial_start = self.net.t

        run_args |= {
            self.fin: fin_all,
            self.fout: fout_all,
            self.ng.t_trial_start: self.net.t,
        }

        if continue_state:
            run_args |= self.continue_state_run_args

        self.net.run(
            T_trial_all, report_period=self._base_params["report_period_s"] * b2.second
        )
        return run_args

    @property
    def continue_state_run_args(self):
        run_args = {}
        for objname, varnames in self._state_vars_to_save.items():
            obj = self.net[objname]
            for varname in varnames:
                run_args[getattr(obj, varname)] = getattr(obj, varname)
        return run_args

    def __reduce__(self):
        net_state = {}
        for attr, state_vars in self._state_vars_to_save.items():
            net_obj = self.__dict__[attr]
            net_state[attr] = {var: getattr(net_obj, var) for var in state_vars}

        return (self.__class__.build, (self._cond_params, self._base_params, net_state))

    @classmethod
    def build(cls, cond_params, base_params, factors, emg, net_state={}) -> "fSNN":
        # lame, but need to import inside here for constructor to work when unpickling
        import numpy as np
        from cleo.coords import assign_coords_rand_rect_prism, coords_from_ng

        from src.helpers import seed_everything

        b2.defaultclock.dt = base_params.dt_ms * b2.ms

        seed = seed_everything(cond_params["seed"], "train_fsnn")
        rng = np.random.default_rng(seed)

        # TODO: calculate volume, num neurons per layer
        thickness = (
            base_params["layer_boundaries_um"][-1]
            - base_params["layer_boundaries_um"][0]
        ) * b2.um
        side_length_um = (
            np.sqrt(base_params["array_side_length_mm"] * 1000) ** 2
            * cond_params["network_area_scale"]
        )
        vol = (side_length_um * b2.um) ** 2 * thickness
        n_ng = vol * (base_params["base_neuron_density"] * b2.mm**-3)
        n_ng = int(round(n_ng))
        print(f"side_length_um: {side_length_um}, volume: {vol}, n_ng: {n_ng}")
        hor_lim = (-side_length_um / 2, side_length_um / 2)

        ############# start fSNN code ##############

        n_out = emg.shape[2]

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

        samp_period = base_params.sample_period_ms * b2.ms
        state_mon = b2.StateMonitor(
            ng, ["X", "Z"], record=np.arange(10), dt=samp_period
        )
        v_mon = b2.StateMonitor(ng, "v", record=np.arange(10))

        net = b2.Network(ng, syn, state_mon, v_mon)

        syn.w = "randn()"

        # timed arrays
        n_tarr = nsteps_large_timed_arrays(base_params, base_params["train_fsnn"])

        fsnn = cls(
            net,
            ng,
            syn,
            state_mon,
            v_mon,
            cond_params["n_pcs"],
            cond_params,
            base_params,
            rng=rng,
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
    # will want a {task: emg|factors} dict
    factors, emg = importlib.import_module("src.fsnn").load_data(cond_params.task)
    factors = {cond_params.task: factors}
    emg = {cond_params.task: emg}

    fsnn = fSNN.build(cond_params, base_params, factors, emg)

    print(f"preparing {base_params.n_trials_demean} demean trials")
    run_args = fsnn.run_trials(
        base_params.n_trials_demean, mode="demean", record=False, continue_state=False
    )
    if train_device == "cuda":
        print("compiling cuda code")
        b2.device.build(
            directory=f"{outdir}/brian2cuda_output",
            results_directory=outdir,
            run=False,
        )
        print("done building")

        b2.device.run(run_args=run_args)

    print(f"preparing {base_params.n_trials_train} training trials")
    run_args = fsnn.run_trials(
        base_params.n_trials_train, mode="train", record=False, continue_state=True
    )
    if train_device == "cuda":
        # run args starts next simulation with state of the end of the previous
        b2.device.run(run_args=run_args)

    print(f"preparing {base_params.n_trials_test} testing trials")
    run_args = fsnn.run_trials(
        base_params.n_trials_test, mode="test", record=True, continue_state=True
    )
    if train_device == "cuda":
        b2.device.run(run_args=run_args)

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
            numerize(snakemake.wildcards),
            snakemake.params,
            snakemake.config["train_device"],
            snakemake.output[0],
        )
