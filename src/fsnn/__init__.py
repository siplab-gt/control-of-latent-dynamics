import importlib
import os
from dataclasses import dataclass, field

import brian2.only as b2
import brian2cuda  # noqa: F401
import numpy as np
from jaxtyping import UInt
from tqdm import tqdm

from src.data_classes import BaseDataClass
from src.fsnn.task import fSNNTask


def run_main(cond_params, base_params, device, fsnn_infile, fsnn_outfile, mode, record):
    outdir = os.path.dirname(fsnn_outfile)
    if device == "cuda":
        b2.set_device("cuda_standalone", build_on_run=False)
    else:
        assert device == "cython"
        b2.prefs.codegen.target = "cython"

    if fsnn_infile is not None:
        print(f"loading fSNN from {fsnn_infile}")
        fsnn = fSNN.load(fsnn_infile)
    else:
        fsnn = fSNN.build(cond_params, base_params)

    n_trials = base_params.n_trials[mode].main
    n_init_trials = base_params.n_trials[mode].init
    n_settle_trials = base_params.n_trials[mode].get("settle", 0)
    print(
        f"Preparing {n_init_trials} warmup trials, {n_trials} {mode} trials"
        f", {n_settle_trials} settle trials"
    )
    fsnn.run_trials(
        n_trials,
        mode=mode,
        record=record,
        n_init_trials=n_init_trials,
        n_settle_trials=n_settle_trials,
    )
    if device == "cuda":
        print("compiling cuda code")
        os.makedirs(f"{outdir}/brian2cuda_output", exist_ok=True)
        b2.device.build(
            directory=f"{outdir}/brian2cuda_output",
            results_directory=outdir,
            run=False,
        )
        print("done building")

        b2.device.run()

    print(f"done running {mode} trials")

    if fsnn_outfile is not None:
        fsnn.save(fsnn_outfile)
        print(f"saved fSNN to {fsnn_outfile}")

    return fsnn


@dataclass
class fSNN(BaseDataClass):
    net: b2.Network
    ytilde_ta: b2.TimedArray
    """target factors"""
    fout_ta: b2.TimedArray
    """target outputs (i.e., EMG)"""
    fin_ta: b2.TimedArray
    """inputs"""
    n_pcs: int
    i_trial_boundaries: UInt[np.ndarray, "n_trials n_events"] = field(
        default_factory=list
    )
    """Trial (start, ..., end) indices, shape (n_trials, n_events) array"""
    conditions: list = field(default_factory=list)
    tasks: list = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    _mode: str = "init"
    _save_data: bool = True
    _cond_params: dict = None
    _base_params: dict = None
    _state_vars_to_save: dict = field(
        default_factory=lambda: dict(
            ng_main=["v", "vJsbar", "ss", "sf", "J0ss", "J0fs", "J0fbars"],
            syn=["w1", "w2", "w3"],
            syn_z2yPlus=["W"],
        )
    )
    _cache: dict = field(repr=False, default_factory=dict)

    def __getitem__(self, key):
        """Allows access to net components by name, e.g. fsnn['ng_yPlus']"""
        try:
            return self.net[key]
        except KeyError as e:
            raise KeyError(f"{key} not found in fSNN network") from e

    def __getattr__(self, name):
        """Delegate attribute access to self.net if not found on self."""
        if name in self.net:
            return self.net[name]
        return super().__getattribute__(name)

    @property
    def k_task(self) -> int:
        return self.ng_in.N

    @property
    def n_in(self) -> int:
        return self.k_task

    @property
    def n_out(self) -> int:
        return self.ng_z.N

    @property
    def n_factors(self) -> int:
        return self.ng_y.N

    def u_task(self, t: b2.Quantity) -> np.ndarray:
        """task-related inputs, shape ([T], n_task_inputs) array"""
        # TimedArray can do multiple timesteps, but not multiple variables
        # so we access from .values ourselves insted of ta(t)
        i = np.astype(t / self.dt, np.int32)
        i = np.clip(i, 0, self.fin_ta.shape[0] - 1)
        return self.fin_ta.values[i]

    @property
    def u_task_(self):
        """callable with signature t -> task-related input"""
        return lambda t: self.u_task(t)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        # enable/disable RLS updates
        for objname in [
            "ng_rand",
            "ng_s",
            "syn_s2rand",
            "syn_main2s",
            "syn_s2s",
            "syn_y2s",
            "ng_yPlus",
            "syn_yPlus2rand",
            "syn_y2yPlus",
            "syn_yPlus2yPlus",
            "syn_z2yPlus",
        ]:
            self.net[objname].active = value == "train"

        if value != "train":
            self.ng_yPlus.k = 0  # besides disabling synapses that update k

        self.ng_y.init_mode = value == "init"

        self._mode = value

    @property
    def monitors(self):
        return [self[name] for name in ["v_mon", "y_mon", "z_mon", "spk_mon"]]

    def _get_namespace(self, t_ta_start):
        ns = {attr: getattr(self, attr) for attr in ["fin_ta", "fout_ta", "ytilde_ta"]}
        ns |= self._base_params  # gets things like tau_s, vth
        ns["t_ta_start"] = t_ta_start
        return ns

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
        dt = self._base_params.dt
        assert dt == b2.defaultclock.dt
        return dt

    def generate_trial(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, tuple[int, int], int, int]:
        """returns ytilde,fout,fin,nsteps_trial,cond,i_trial_start_end,ITIb,ITI"""
        task = self.rng.choice(self.tasks)
        return task.generate_trial(self.rng)

    def run_trial(self, trial_params=None):
        if trial_params is None:
            ytilde, fout, fin, nsteps_trial, condition, i_trial_bounds, nsteps_iti = (
                self.generate_trial()
            )
        else:
            ytilde, fout, fin, nsteps_trial, condition, i_trial_bounds, nsteps_iti = (
                trial_params
            )
        T_trial = nsteps_trial * self.dt
        self.fin_ta.values = fin
        self.fout_ta.values = fout
        self.ytilde_ta.values = ytilde
        self.net.run(
            T_trial,
            report_period=self._base_params["report_period_s"] * b2.second,
            # need this so TimedArrays can work with respect to trial start
            namespace=self._get_namespace(t_ta_start=self.net.t),
        )

        if self._save_data:
            self.conditions.append(condition)
            self.i_trial_boundaries.append(i_trial_bounds)

    def run_trials(
        self, n_main_trials, mode, record=False, n_init_trials=50, n_settle_trials=0
    ):
        """needed for CUDA standalone training: mainly to prep TimedArrays
        with all trials in advance"""

        self.mode = "init"

        fin_all = np.zeros((0, self.n_in))
        fout_all = np.zeros((0, self.n_out))
        ytilde_all = np.zeros((0, self.n_factors))

        T = {}
        nsteps_prev = 0
        # must be single giant timed arrays for multiple run calls
        for _mode, n_trials in [
            ("init", n_init_trials),
            (mode, n_main_trials),
            ("init", n_settle_trials),
        ]:
            print(f"Preparing {_mode} trials: {n_trials} trials")
            for i in tqdm(range(n_trials)):
                (
                    ytilde,
                    fout,
                    fin,
                    nsteps_trial,
                    condition,
                    i_trial_bounds,
                    nsteps_iti,
                ) = self.generate_trial()
                T_trial = nsteps_trial * self.dt
                fin_all = np.vstack([fin_all, fin])
                fout_all = np.vstack([fout_all, fout])
                ytilde_all = np.vstack([ytilde_all, ytilde])
                T[_mode] = T.get(_mode, 0) + T_trial
                if self._save_data and _mode == mode:
                    self.conditions.append(condition)
                    i_start = i_trial_bounds[0] + nsteps_prev
                    i_end = i_trial_bounds[1] + nsteps_prev
                    self.i_trial_boundaries.append((i_start, i_end))
                    # only index data for saved trials (i.e., not for init)
                    nsteps_prev += nsteps_trial

        self.fin_ta.values = fin_all
        self.fout_ta.values = fout_all
        self.ytilde_ta.values = ytilde_all

        for _mode in ["init", mode, "init"]:
            self.mode = _mode
            self.net.run(
                T[_mode],
                report_period=self._base_params.report_period,
                namespace=self._get_namespace(t_ta_start=0 * b2.ms),
            )

    def __reduce__(self):
        net_state = {}
        for attr, state_vars in self._state_vars_to_save.items():
            net_obj = self[attr]
            net_state[attr] = {var: getattr(net_obj, var) for var in state_vars}

        return (
            self.__class__.build,
            (self._cond_params, self._base_params, net_state, self._cache),
        )

    @classmethod
    def build(cls, cond_params, base_params, net_state={}, cache={}) -> "fSNN":
        # lame, but need to import inside here for constructor to work when unpickling
        import numpy as np
        from cleo.coords import assign_coords_rand_rect_prism, coords_from_ng

        from src.helpers import seed_everything

        print("building fSNN")
        print(f"cond_params: {cond_params}")
        print(f"base_params: {base_params}")

        dt = base_params.dt
        b2.defaultclock.dt = dt

        seed = seed_everything(cond_params["seed"], "train_fsnn")
        rng = np.random.default_rng(seed)

        # TODO: calculate volume, num neurons per layer
        thickness = base_params.layer_boundaries[-1] - base_params.layer_boundaries[0]
        side_length_um = (
            base_params.array_side_length * np.sqrt(cond_params.network_area_scale)
        ) / b2.um
        vol = (side_length_um * b2.um) ** 2 * thickness
        n_ng = vol * (base_params["base_neuron_density"] * b2.mm**-3)
        n_ng = int(round(n_ng))
        print(f"side_length_um: {side_length_um}, volume: {vol}, n_ng: {n_ng}")
        hor_lim = (-side_length_um / 2, side_length_um / 2)

        ############# start fSNN code ##############

        if cond_params.get("multitask", False):
            task_names = ["reaching", "cycling"]
            weight_params = base_params
        else:
            task_names = [cond_params.task]
            weight_params = base_params | base_params[cond_params.task]

        # task cache init since it can be slow (esp. PCA trials of reaching rate network)
        if "tasks" in cache:
            tasks = cache["tasks"]
        else:
            tasks: list[fSNNTask] = [
                importlib.import_module(f"src.fsnn.{task}").init_task(
                    seed,
                    base_params,
                    base_params[cond_params.task],
                    n_ng,
                    weight_params.g,
                )
                for task in task_names
            ]

        n_out = sum([task.n_out for task in tasks])
        n_in = sum([task.n_in for task in tasks])
        n_factors = sum([task.n_factors for task in tasks])

        u = np.hstack([task.u for task in tasks])
        u_in = np.hstack([task.u_in for task in tasks])

        fin_ta = b2.TimedArray(np.zeros((1, n_in)), dt=dt)
        ng_in = b2.NeuronGroup(
            N=n_in,
            model="value = fin_ta(t - t_ta_start, i) : 1",
            name="ng_in",
        )

        ng_main = b2.NeuronGroup(
            N=n_ng,
            model="""
            vJsbar : 1
            uy : 1
            I_in : 1
            vinf = J0fbars - vJsbar + uy + J0ss + J0fs + I_in : 1
            dv/dt = (v - vinf) * (-1 / tau_v) : 1
            dJ0ss/dt = -J0ss/tau_s : 1
            dJ0fs/dt = -J0fs/tau_f : 1
            dJ0fbars/dt = -J0fbars/tau_f : 1
            dss/dt = -ss/tau_s : 1
            dsf/dt = -sf/tau_f : 1
            sum_vJsbar : 1
            """,
            method="euler",
            threshold="v>=vth",
            reset="""
            v=vr
            ss += 1
            sf += 1
            """,
            name="ng_main",
        )
        ng_main.v = 1e-3 * rng.normal(size=n_ng)

        assign_coords_rand_rect_prism(
            ng_main,
            hor_lim,
            hor_lim,
            (
                base_params.layer_boundaries[0] / b2.um,
                base_params.layer_boundaries[-1] / b2.um,
            ),
            unit=b2.um,
        )
        print(f"{ng_main.N=}")

        syn = b2.Synapses(
            ng_main,
            ng_main,
            """ w1 : 1
                w2 : 1
                w3 : 1
            """,
            on_pre="""
               J0ss += w1
               J0fs += w2
               J0fbars += w3
            """,
            name="syn",
        )
        syn.connect()

        tau_s = base_params.tau_s
        tau_f = base_params.tau_f
        syn.w1 = (
            weight_params.gs
            / (b2.second / tau_s)
            / np.sqrt(n_ng)
            * rng.normal(size=n_ng * n_ng)
        )  # J0s
        syn.w2 = (
            weight_params.gf
            / (b2.second / tau_f)
            / np.sqrt(n_ng)
            * rng.normal(size=n_ng * n_ng)
        )  # J0f
        syn.w3 = weight_params.muf / (b2.second / tau_f) / n_ng  # Jmu

        # these are just placeholders, will set values when running trials
        ytilde_ta = b2.TimedArray(np.zeros((1, n_factors)), dt=dt)
        fout_ta = b2.TimedArray(np.zeros((1, n_out)), dt=dt)

        syn_in = b2.Synapses(
            ng_in,
            ng_main,
            """u_in : 1 (constant)
            I_in_post = value_pre * u_in : 1 (summed)""",
            name="syn_in",
        )
        syn_in.connect()
        # I think u_in should be transposed first, since brian uses i, j to refer to pre, post
        assert u_in.shape == (n_ng, n_in)
        syn_in.u_in = u_in.T.flatten()

        ng_z = b2.NeuronGroup(
            N=n_out,
            model="""
            fout = fout_ta(t - t_ta_start, i) : 1
            z : 1
            """,
            name="ng_z",
        )

        ng_y = b2.NeuronGroup(
            N=n_factors,
            model="""
            ytilde = ytilde_ta(t - t_ta_start, i) : 1
            y : 1
            y_mon = ytilde * int(init_mode) + y * (1 - int(init_mode)) : 1
            err = y_mon - ytilde : 1
            init_mode : boolean (shared)
            """,
            name="ng_y",
        )

        # ng_nMSE = b2.NeuronGroup(
        #     N=1,
        #     model="""
        #     nMSE_numerator : 1
        #     nMSE_denominator : 1
        #     nMSE_sum : 1
        #     nMSE : 1
        #     """,
        #     name="ng_nMSE",
        # )
        # ng_nMSE.run_regularly(
        #     """
        #     nMSE_sum += nMSE_numerator / (nMSE_denominator + 1e-9)
        #     """,
        #     dt=dt,
        #     name="ng_nMSE_run_regularly_dt",
        # )
        # ng_nMSE.run_regularly(
        #     """
        #     nMSE = nMSE_sum / 2000
        #     """,
        #     dt=dt * 2000,
        #     name="ng_nMSE_run_regularly_dTTrial",
        # )
        # # NOTE: with hard-coded dt this won't be exactly every trial for reaching
        # synapses_y2nMSE = b2.Synapses(
        #     ng_y,
        #     ng_nMSE,
        #     model="""
        #     nMSE_numerator_post = err_pre ** 2 : 1 (summed)
        #     nMSE_denominator_post = ytilde_pre ** 2 : 1 (summed)
        #     """,
        #     name="syn_y2nMSE",
        # )
        # synapses_y2nMSE.connect()
        # statemon_nMSE = b2.StateMonitor(
        #     ng_nMSE, "nMSE", record=[0], name="statemonitor_nMSE"
        # )
        # NOTE: will forget about simulating this for now and simply calculate post hoc

        # To check
        syn_uy = b2.Synapses(
            ng_y,
            ng_main,
            model="""
            u : 1 (constant)
            uy_post = u * y_mon_pre : 1 (summed)
            """,
            name="syn_uy",
        )

        assert u.shape == (n_ng, n_factors)
        syn_uy.connect()
        syn_uy.u = u.T.flatten()

        ####### RLS ########

        neurons_rand = b2.NeuronGroup(
            N=1,
            model="""
            randSingle = rand() : 1 (constant over dt)
            """,
            threshold="randSingle < 1.0 / DTRLS",
            name="ng_rand",
        )
        ng_s = b2.NeuronGroup(
            n_ng,
            model="""
                                ss : 1
                                sf : 1
                                xPUp_tot : 1
                                xPDown_tot : 1
                                kdm1 : 1
                                kLeft : 1
                                kRight : 1
                                rand_trigger : 1
                                """,
            threshold="rand_trigger < 1.0 / DTRLS",
            name="ng_s",
        )
        syn_s2rand = b2.Synapses(
            ng_s,
            neurons_rand,
            model="""
            rand_trigger_pre = randSingle_post : 1 (summed)
            """,
            name="syn_s2rand",
        )
        syn_s2rand.connect()
        syn_main2s = b2.Synapses(
            ng_main,
            ng_s,
            model="""
            ss_post = ss_pre : 1 (summed)
            sf_post = sf_pre : 1 (summed)
            """,
            name="syn_main2s",
        )
        syn_main2s.connect(condition="i==j")

        # connecting neurons_s and itself
        synapses_s2s = b2.Synapses(
            ng_s,
            ng_s,
            model="""
                                xPUp : 1
                                xPDown : 1

                                xPUp_tot_pre = xPUp : 1 (summed)
                                xPDown_tot_pre = xPDown : 1 (summed)

                                kdm1_pre = ss_post * xPUp_tot_post + sf_post * xPDown_tot_post : 1 (summed)

                                Pw_A : 1
                                Pw_B : 1
                                Pw_C : 1
                                Pw_D : 1
                                """,
            on_pre={
                "pre_step1_xP": """
                                    xPUp = Pw_A * ss_post + Pw_B * sf_post
                                    xPDown = Pw_C * ss_post + Pw_D * sf_post
                                    """,
                "pre_step2_k": """
                                    kLeft_pre = xPUp_tot_pre / (1 + kdm1_pre)
                                    kRight_pre = xPDown_tot_pre / (1 + kdm1_pre)
                                    """,
                "pre_step3_Pw": """
                                    Pw_A -= xPUp_tot_pre * kLeft_post
                                    Pw_B -= xPUp_tot_pre * kRight_post
                                    Pw_C -= xPDown_tot_pre * kLeft_post
                                    Pw_D -= xPDown_tot_pre * kRight_post
                                    """,
            },
            name="syn_s2s",
        )
        synapses_s2s.pre_step1_xP.when = "synapses"
        synapses_s2s.pre_step1_xP.order = -5

        synapses_s2s.summed_updaters["xPUp_tot_pre"].when = "synapses"
        synapses_s2s.summed_updaters["xPUp_tot_pre"].order = (
            synapses_s2s.pre_step1_xP.order + 1
        )
        synapses_s2s.summed_updaters["xPDown_tot_pre"].when = "synapses"
        synapses_s2s.summed_updaters["xPDown_tot_pre"].order = (
            synapses_s2s.pre_step1_xP.order + 1
        )

        synapses_s2s.summed_updaters["kdm1_pre"].when = "synapses"
        synapses_s2s.summed_updaters["kdm1_pre"].order = (
            synapses_s2s.summed_updaters["xPUp_tot_pre"].order + 1
        )

        synapses_s2s.pre_step2_k.when = "synapses"
        synapses_s2s.pre_step2_k.order = (
            synapses_s2s.summed_updaters["kdm1_pre"].order + 1
        )

        synapses_s2s.pre_step3_Pw.when = "synapses"
        synapses_s2s.pre_step3_Pw.order = synapses_s2s.pre_step2_k.order + 1

        synapses_s2s.connect()
        synapses_s2s.Pw_A = 1.0 * np.eye(n_ng).flatten()  # inverse covariance matrix
        synapses_s2s.Pw_B = 1.0 * np.eye(n_ng).flatten()  # inverse covariance matrix
        synapses_s2s.Pw_C = 1.0 * np.eye(n_ng).flatten()  # inverse covariance matrix
        synapses_s2s.Pw_D = 1.0 * np.eye(n_ng).flatten()  # inverse covariance matrix

        # connecting neurons_s and neurons_y
        synapses_y2s = b2.Synapses(
            ng_y,
            ng_s,
            model="""
            wLeft : 1
            wRight : 1
            y_pre = wLeft * ss_post + wRight * sf_post : 1 (summed)
            """,
            on_post="""
            wLeft -= (y_pre - ytilde_pre) * kLeft_post
            wRight -= (y_pre - ytilde_pre) * kRight_post
            """,
            name="syn_y2s",
        )
        synapses_y2s.connect()

        ng_yPlus = b2.NeuronGroup(
            N=n_factors + 1,
            model="""
            y : 1
            yPlus = y * int(i<N-1) + 1.0 * int(i==N-1) : 1
            xP_tot : 1
            kdm1 : 1
            k : 1
            rand_trigger : 1
            """,
            threshold="rand_trigger < 1.0 / DTRLS",
            reset="",
            name="ng_yPlus",
        )
        syn_yPlus2rand = b2.Synapses(
            ng_yPlus,
            neurons_rand,
            model="""
            rand_trigger_pre = randSingle_post : 1 (summed)
            """,
            name="syn_yPlus2rand",
        )
        syn_yPlus2rand.connect()
        syn_y2yPlus = b2.Synapses(
            ng_y,
            ng_yPlus,
            model="""
            y_post = y_pre : 1 (summed)
            """,
            name="syn_y2yPlus",
        )
        syn_y2yPlus.connect(condition="i == j", skip_if_invalid=True)

        # connecting yPlus and itself
        syn_yPlus2yPlus = b2.Synapses(
            ng_yPlus,
            ng_yPlus,
            model="""
            xP : 1
            xP_tot_pre = xP : 1 (summed)
            kdm1_pre = yPlus_post * xP_tot_post : 1 (summed)
            PW : 1
            """,
            on_pre={
                "pre_step1_xP": "xP = PW * yPlus_post",
                "pre_step2_k": "k_pre = xP_tot_pre / (1 + kdm1_pre)",
                "pre_step3_PW": "PW -= xP_tot_pre * k_post",
            },
            name="syn_yPlus2yPlus",
        )
        syn_yPlus2yPlus.connect()
        syn_yPlus2yPlus.PW = 1.0 * np.eye(n_factors + 1).flatten()

        syn_yPlus2yPlus.pre_step1_xP.when = "synapses"
        syn_yPlus2yPlus.pre_step1_xP.order = -5

        syn_yPlus2yPlus.summed_updaters["xP_tot_pre"].when = "synapses"
        syn_yPlus2yPlus.summed_updaters["xP_tot_pre"].order = (
            syn_yPlus2yPlus.pre_step1_xP.order + 1
        )

        syn_yPlus2yPlus.summed_updaters["kdm1_pre"].when = "synapses"
        syn_yPlus2yPlus.summed_updaters["kdm1_pre"].order = (
            syn_yPlus2yPlus.summed_updaters["xP_tot_pre"].order + 1
        )

        syn_yPlus2yPlus.pre_step2_k.when = "synapses"
        syn_yPlus2yPlus.pre_step2_k.order = (
            syn_yPlus2yPlus.summed_updaters["kdm1_pre"].order + 1
        )

        syn_yPlus2yPlus.pre_step3_PW.when = "synapses"
        syn_yPlus2yPlus.pre_step3_PW.order = syn_yPlus2yPlus.pre_step2_k.order + 1

        # connecting yPlus and z
        syn_z2yPlus = b2.Synapses(
            ng_z,
            ng_yPlus,
            model="""
            W : 1
            z_pre = W * yPlus_post : 1 (summed)
            """,
            on_post="""
            W -= (z_pre - f_pre) * k_post
            """,
            name="syn_z2yPlus",
        )
        syn_z2yPlus.connect()

        samp_period = base_params.sample_period
        y_mon = b2.StateMonitor(
            ng_y, ["y_mon", "ytilde"], record=True, name="y_mon", dt=samp_period
        )
        z_mon = b2.StateMonitor(
            ng_z, ["fout"], record=True, name="z_mon", dt=samp_period
        )

        ############# end code derived from Yijun's reaching_b2c.py ##############

        v_mon = b2.StateMonitor(ng_main, "v", record=np.arange(10))
        spk_mon = b2.SpikeMonitor(ng_main, name="spk_mon")

        net = b2.Network(b2.collect())

        fsnn = cls(
            net,
            ytilde_ta,
            fout_ta,
            fin_ta,
            cond_params["n_pcs"],
            _cond_params=cond_params,
            _base_params=base_params,
            rng=rng,
            tasks=tasks,
            _cache=cache,
        )

        for attr, state_vars in net_state.items():
            net_obj = getattr(fsnn, attr)
            print(f"loading state for {attr}: {list(state_vars.keys())}")
            for var, values in state_vars.items():
                getattr(net_obj, var)[:] = values

        print(f"{coords_from_ng(ng_main)[:10]=}")
        return fsnn
