import importlib
import logging
from dataclasses import dataclass, field

import brian2.only as b2
from brian2 import np
from jaxtyping import UInt
from tqdm import tqdm

from src.data_classes import BaseDataClass
from src.fsnn.task import fSNNTask
from src.helpers import save_pkl_gz, setup_debugger

log = logging.getLogger(__name__)


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
    _record: bool = False
    _save_data: bool = True
    _cond_params: dict = None
    _base_params: dict = None
    _state_vars_to_save: dict = field(
        default_factory=lambda: dict(
            ng_main=["v", "vJsbar", "ss", "sf", "J0ss", "J0fs", "J0fbars"],
            syn_y2s=["wLeft", "wRight"],
            syn_z2yPlus=["W"],
        )
    )
    _cache: dict = field(repr=False, default_factory=dict)
    _t_prev: b2.Quantity = field(default_factory=lambda: 0 * b2.ms, repr=False)
    _i_record_prev: int = 0
    _gen_trial_state: dict = field(default_factory=dict, repr=False)

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
        # settle is an alias for init, for running after main mode
        if value == "settle":
            value = "init"
        log.debug(f"setting fSNN mode to {value}")

        # only update vJsbar during demean
        self.vJsbar_update.active = value == "demean"

        # enable/disable RLS updates
        for objname in [
            "ng_rls_trigger",
            "ng_s",
            "syn_main2s",
            "syn_s2s",
            "syn_s2kdm1",
            "syn_y2s",
            "ng_yPlus",
            "syn_y2yPlus",
            "syn_yPlus2yPlus",
            "syn_z2yPlus",
        ]:
            self.net[objname].active = value == "train"

        if value != "train":
            self.ng_yPlus.k = 0  # besides disabling synapses that update k

        self.ng_y.target_feedback_mode = value in ["init", "demean"]

        self._mode = value

    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, value):
        self._record = value
        for mon in self.monitors:
            mon.active = value

    @property
    def monitors(self):
        return (
            obj
            for obj in self.net.sorted_objects
            if isinstance(obj, (b2.StateMonitor, b2.SpikeMonitor))
        )

    def save_mon_data(self, filepath):
        data = {}
        for mon in self.monitors:
            data[mon.name] = mon.get_states()
        save_pkl_gz(data, filepath)

    def _get_namespace(self, t_ta_start):
        ns = {attr: getattr(self, attr) for attr in ["fin_ta", "fout_ta", "ytilde_ta"]}
        ns |= self._base_params  # gets things like tau_s, vth
        ns["t_ta_start"] = t_ta_start
        ns["sum"] = sum
        return ns

    @property
    def dt(self) -> b2.Quantity:
        dt = self._base_params.dt
        assert dt == b2.defaultclock.dt
        return dt

    def generate_trial(
        self, **kwargs
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, int, int, tuple[int, int], int, int, dict
    ]:
        """returns ytilde,fout,fin,nsteps_trial,cond,i_trial_start_end,ITIb,ITI,kwargs"""
        task = self.rng.choice(self.tasks)
        return task.generate_trial(self.rng, **kwargs)

    def run_trial(self, record=False):
        """run a single trial of the current mode"""
        # relies on object state to keep track of task state, time, nsteps
        self.record = record
        (
            ytilde,
            fout,
            fin,
            nsteps_trial,
            condition,
            i_trial_bounds,
            nsteps_iti,
            self._gen_trial_state,  # e.g., x for rate network state
        ) = self.generate_trial(**self._gen_trial_state)

        T_trial = nsteps_trial * self.dt
        self.fin_ta = b2.TimedArray(fin, dt=self.dt)
        self.fout_ta = b2.TimedArray(fout, dt=self.dt)
        self.ytilde_ta = b2.TimedArray(ytilde, dt=self.dt)
        self.net.run(
            T_trial,
            report="stdout",
            report_period=self._base_params.report_period,
            # need this so TimedArrays can work with respect to trial start
            namespace=self._get_namespace(t_ta_start=self._t_prev),
        )
        self._t_prev += T_trial

        if record:
            self.conditions.append(condition)
            self.i_trial_boundaries.append(
                (i + self._i_record_prev for i in i_trial_bounds)
            )
            self._i_record_prev += nsteps_trial

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
        phases = [
            ("init", n_init_trials, True),  # TODO; false
            (mode, n_main_trials, record),
            ("settle", n_settle_trials, False),
        ]
        # must be single giant timed arrays for multiple run calls
        gen_trial_state = {}
        for _mode, n_trials, _record in phases:
            self.record = _record
            self.mode = _mode
            print(f"Preparing {n_trials} {_mode} trials:")
            for i in tqdm(range(n_trials)):
                (
                    ytilde,
                    fout,
                    fin,
                    nsteps_trial,
                    condition,
                    i_trial_bounds,
                    nsteps_iti,
                    gen_trial_state,
                ) = self.generate_trial(**gen_trial_state)
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

        self.fin_ta = b2.TimedArray(fin_all, dt=self.dt)
        self.fout_ta = b2.TimedArray(fout_all, dt=self.dt)
        self.ytilde_ta = b2.TimedArray(ytilde_all, dt=self.dt)
        assert self.ytilde_ta.values.shape[0] > 0
        assert not np.all(self.ytilde_ta.values == 0), "ytilde_ta is all zeros!"

        for _mode, n_trials, _record in phases:
            self.mode = _mode
            assert self.syn_s2s.pre_step2_k.active == (_mode == "train")
            self.record = _record
            log.info(f"calling net.run() for {T.get(_mode, 0 * b2.ms)}")
            self.net.run(
                T.get(_mode, 0 * b2.ms),
                report="stdout",
                report_period=self._base_params.report_period,
                namespace=self._get_namespace(t_ta_start=0 * b2.ms),
                profile=(_mode == mode),
            )
        # setup_debugger()

    def __reduce__(self):
        net_state = {}
        for obj_name, state_vars in self._state_vars_to_save.items():
            net_obj = self[obj_name]
            net_state[obj_name] = net_obj.get_states(state_vars)

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
        b2.prefs.core.default_float_dtype = getattr(np, base_params.default_float_dtype)

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
            log.info("using cached tasks")
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
            cache["tasks"] = tasks

        n_out = sum([task.n_out for task in tasks])
        n_in = sum([task.n_in for task in tasks])
        n_factors = sum([task.n_factors for task in tasks])

        u = np.hstack([task.u for task in tasks])
        u_in = np.hstack([task.u_in for task in tasks])

        fin_ta = b2.TimedArray(np.ones((1, n_in)), dt=dt)
        ng_in = b2.NeuronGroup(
            N=n_in,
            model="fin = fin_ta(t - t_ta_start, i) : 1",
            name="ng_in",
        )

        ng_main = b2.NeuronGroup(
            N=n_ng,
            model="""
            vJsbar : 1
            n_t : 1 (shared)  # for computing vJsbar
            uy : 1
            I_in : 1
            vinf = J0fbars - vJsbar + uy + J0ss + J0fs + I_in : 1
            dv/dt = (v - vinf) * (-1 / tau_v) : 1
            dJ0ss/dt = -J0ss/tau_s : 1
            dJ0fs/dt = -J0fs/tau_f : 1
            dJ0fbars/dt = -J0fbars/tau_f : 1
            dss/dt = -ss/tau_s : 1
            dsf/dt = -sf/tau_f : 1
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
        ng_main.run_regularly(
            """ n_t += 1
                vJsbar += (uy + J0ss + J0fs - vJsbar) / n_t""",
            name="vJsbar_update",
            dt=dt,
            when="end",
        )
        log.debug(f"{ng_main.v.dtype=}")
        ng_main.v = 1e-3 * rng.normal(size=n_ng)
        log.debug(f"  and after initialization: {ng_main.v.dtype=}")

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

        syn_main = b2.Synapses(
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
            name="syn_main",
        )
        syn_main.connect()

        tau_s = base_params.tau_s
        tau_f = base_params.tau_f
        syn_main.w1 = (
            weight_params.gs
            / (b2.second / tau_s)
            / np.sqrt(n_ng)
            * rng.normal(size=n_ng * n_ng)
        )  # J0s
        syn_main.w2 = (
            weight_params.gf
            / (b2.second / tau_f)
            / np.sqrt(n_ng)
            * rng.normal(size=n_ng * n_ng)
        )  # J0f
        syn_main.w3 = weight_params.muf / (b2.second / tau_f) / n_ng  # Jmu

        # these are just placeholders, will re-assign when running trials
        ytilde_ta = b2.TimedArray(np.ones((1, n_factors)), dt=dt)
        fout_ta = b2.TimedArray(np.ones((1, n_out)), dt=dt)

        syn_in = b2.Synapses(
            ng_in,
            ng_main,
            """u_in : 1 (constant)
            I_in_post = fin_pre * u_in : 1 (summed)""",
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
            # y_ assumes target value for init and demean modes
            y_ = ytilde * int(target_feedback_mode) + y * (1 - int(target_feedback_mode)) : 1
            target_feedback_mode : boolean (shared)
            """,
            name="ng_y",
        )

        # To check
        syn_uy = b2.Synapses(
            ng_y,
            ng_main,
            model="""
            u : 1 (constant)
            uy_post = u * y__pre : 1 (summed)
            """,
            name="syn_uy",
        )

        assert u.shape == (n_ng, n_factors)
        syn_uy.connect()
        syn_uy.u = u.T.flatten()

        ####### RLS ########

        ng_rls_trigger = b2.NeuronGroup(
            N=1,
            model="""
            rls_trigger = rand() : 1 (constant over dt)
            """,
            name="ng_rls_trigger",
        )
        ng_s = b2.NeuronGroup(
            n_ng,
            model="""
                ss : 1
                sf : 1
                xPUp : 1
                xPDown : 1
                # will update in run_regularly to be able to customize schedule
                kdm1_per_neuron : 1   # = ss * xPUp + sf * xPDown : 1
                kdm1_linked : 1 (linked)
                # stupid, but need to copy here so synapses can see it
                # synapses can't have the linked var itself
                kdm1 = kdm1_linked : 1 (shared)
                kLeft : 1
                kRight : 1
                rls_trigger : 1 (linked)
            """,
            threshold="rls_trigger < 1.0 / dtrls",
            reset="kdm1_per_neuron = ss * xPUp + sf * xPDown",  # only on RLS; must be scheduled (not default resets phase)
            name="ng_s",
        )
        ng_s.rls_trigger = b2.linked_var(ng_rls_trigger, "rls_trigger")
        ng_kdm1 = b2.NeuronGroup(1, "kdm1 : 1", name="ng_kdm1")
        ng_s.kdm1_linked = b2.linked_var(ng_kdm1, "kdm1")

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
        syn_s2s = b2.Synapses(
            ng_s,
            ng_s,
            model="""
                # NOTE: I'd like to compute these in on_pre, but can't do summed vars there
                xPUp_pre = Pw_A * ss_post + Pw_B * sf_post : 1 (summed)
                xPDown_pre = Pw_C * ss_post + Pw_D * sf_post : 1 (summed)

                Pw_A : 1
                Pw_B : 1
                Pw_C : 1
                Pw_D : 1
            """,
            on_pre={
                # can't do (summed) here.
                # "pre_step1_xP": """
                #     kdm1_per_neuron_pre = ss_post * xPUp_post + sf_post * xPDown_post
                #     xPUp_pre = Pw_A * ss_post + Pw_B * sf_post
                #     xPDown_pre = Pw_C * ss_post + Pw_D * sf_post
                #     """,
                "pre_step2_k": """
                    kLeft_pre = xPUp_pre / (1 + kdm1_pre)
                    kRight_pre = xPDown_pre / (1 + kdm1_pre)
                    """,
                "pre_step3_Pw": """
                    Pw_A -= xPUp_pre * kLeft_post
                    Pw_B -= xPUp_pre * kRight_post
                    Pw_C -= xPDown_pre * kLeft_post
                    Pw_D -= xPDown_pre * kRight_post
                    """,
            },
            name="syn_s2s",
        )
        syn_s2kdm1 = b2.Synapses(
            ng_s,
            ng_kdm1,
            model="kdm1_post = kdm1_per_neuron_pre : 1 (summed)",
            name="syn_s2kdm1",
        )
        syn_s2kdm1.connect()

        syn_s2s.connect()
        syn_s2s.Pw_A = 1.0 * np.eye(n_ng).flatten()  # inverse covariance matrix
        # syn_s2s.Pw_B = 1.0 * np.eye(n_ng).flatten()  # inverse covariance matrix
        # syn_s2s.Pw_C = 1.0 * np.eye(n_ng).flatten()  # inverse covariance matrix
        syn_s2s.Pw_D = 1.0 * np.eye(n_ng).flatten()  # inverse covariance matrix

        # connecting neurons_s and neurons_y
        syn_y2s = b2.Synapses(
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
        syn_y2s.connect()

        order = -5

        def schedule(updater, update_order=True):
            nonlocal order
            updater.when = "synapses"
            updater.order = order
            if update_order:
                order += 1

        # schedule(syn_s2s.pre_step1_xP)
        schedule(syn_s2s.summed_updaters["xPUp_pre"], False)
        schedule(syn_s2s.summed_updaters["xPDown_pre"])

        schedule(ng_s.resetter["spike"])  # kdm1_per_neuron
        schedule(syn_s2kdm1.summed_updaters["kdm1_post"])
        schedule(syn_s2s.pre_step2_k)
        schedule(syn_s2s.pre_step3_Pw)
        schedule(syn_y2s.post)

        # NOTE: this could be combined with ng_y, would just need to make
        # u not connect from bias neuron
        ng_yPlus = b2.NeuronGroup(
            N=n_factors + 1,
            model="""
            y : 1
            yPlus = y * int(i<N-1) + 1.0 * int(i==N-1) : 1
            xP_tot : 1
            kdm1 : 1
            k : 1
            rls_trigger : 1 (linked)
            """,
            threshold="rls_trigger < 1.0 / dtrls",
            reset="",
            name="ng_yPlus",
        )
        ng_yPlus.rls_trigger = b2.linked_var(ng_rls_trigger, "rls_trigger")
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
            W -= (z_pre - fout_pre) * k_post
            """,
            name="syn_z2yPlus",
        )
        syn_z2yPlus.connect()

        samp_period = base_params.sample_period
        y_mon = b2.StateMonitor(
            ng_y, ["y_", "ytilde"], record=True, name="y_mon", dt=samp_period
        )
        z_mon = b2.StateMonitor(
            ng_z, ["z", "fout"], record=True, name="z_mon", dt=samp_period
        )
        in_mon = b2.StateMonitor(
            ng_in, "fin", record=True, name="in_mon", dt=samp_period
        )
        w_mon = b2.StateMonitor(
            syn_y2s, ["wLeft", "wRight"], record=np.arange(5), name="w_mon"
        )
        W_mon = b2.StateMonitor(syn_z2yPlus, "W", record=np.arange(5), name="W_mon")
        s_mon = b2.StateMonitor(
            ng_s,
            ["kLeft", "kRight", "kdm1_per_neuron"],
            record=np.arange(5),
            name="s_mon",
        )
        kdm1_mon = b2.StateMonitor(ng_kdm1, "kdm1", record=True, name="kdm1_mon")

        ############# end code derived from Yijun's reaching_b2c.py ##############

        main_mon = b2.StateMonitor(
            ng_main,
            [
                "v",
                "vJsbar",
                # "ss",
                # "sf",
                # "J0ss",
                # "J0fs",
            ],
            record=np.arange(5),
            name="main_mon",
        )
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

        log.debug(f"{coords_from_ng(ng_main)[:10]=}")
        return fsnn
