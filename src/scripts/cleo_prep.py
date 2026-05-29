import logging
from dataclasses import dataclass, field
from typing import Self

import brian2.only as b2
import cleo
import numpy as np

from src.data_classes import BaseDataClass
from src.fsnn import fSNN
from src.helpers import PassthroughIOProcessor, preprocess_params, setup_debugger, setup_logging

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
            # separate files so one can change without causing the other to be rerun.
            # Multiple settings can share a cleo_prep module — only the array
            # layout is keyed off `setting`, not the model-fit method or opsin
            # use. The dict below collapses that aliasing so we don't ship
            # one-line stub modules for every aliased setting.
            setting = cond_params.setting
            cleo_prep_module_for = {
                "sufficient_stim": "sufficient_stim",
                "optimal_stim_rec": "optimal_stim_rec",
                "usid_stim_rec": "usid_stim_rec",
                "usid_stim_rec_no_opsin": "usid_stim_rec",
                "braid_stim_rec": "usid_stim_rec",
                "braid_stim_rec_nonlinear": "usid_stim_rec",
            }
            module_name = f"src.cleo_prep_{cleo_prep_module_for[setting]}"
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

        # Opsin dispatch — inlined (not a module-level helper) so it survives
        # dill/snakemake's tmp-script pickle round-trip without losing
        # symbol resolution.
        opsin_class = getattr(base_params, "opsin_plant_class", "proportional_current")
        if opsin_class == "proportional_current":
            # tune opsin strength relative to input weights:
            # 5 mW/mm2 at 0.5 mm distance ≈ typical task input (~0.1 unitless)
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
        elif opsin_class == "chr2_4s_fsnn":
            # fSNN-dimensionless Bansal 4-state ChR2 (UnitlessChr2FourState).
            # Params from results/opsin_calibration/chr2_4s_fsnn_calibration.json
            # produced by rule calibrate_opsin_plant. Imports are local
            # because CleoPrep.__reduce__ re-invokes build() under dill
            # from downstream rules, whose __main__ namespace doesn't
            # re-execute cleo_prep.py's module-level imports.
            import json
            from pathlib import Path
            from src.opsins import UnitlessChr2FourState

            calib_dir = Path(getattr(
                base_params, "opsin_calibration_dir", "results/opsin_calibration"
            ))
            calib_path = calib_dir / "chr2_4s_fsnn_calibration.json"
            with calib_path.open() as f:
                calib = json.load(f)

            # Schema validation: required fields + variant guard. Bare json.load()
            # + dict access masks stale calibrations from the pre-Bansal era and
            # silently produces wrong values if a key is missing or if a future
            # schema change adds an unrecognized field that callers should review.
            _required = {
                "g0", "E", "v_median", "target_iopto", "target_light_mWmm2",
                "saturation_light_mWmm2", "model_period_ms", "fphi_ss_at_target",
                "v_minus_E_at_v_median", "bench", "voperating_ref",
                "calibration_hash", "variant",
            }
            _allowed = _required | {
                "window", "v_low", "v_high", "pulse_duration_ms",
                "bench_duration_ms", "ratios", "cleo_class",
            }
            missing = _required - calib.keys()
            if missing:
                raise ValueError(
                    f"Calibration JSON {calib_path} is missing required field(s): "
                    f"{sorted(missing)}. Re-run rule calibrate_opsin_plant."
                )
            unknown = calib.keys() - _allowed
            if unknown:
                raise ValueError(
                    f"Calibration JSON {calib_path} has unknown field(s): "
                    f"{sorted(unknown)}. If the schema legitimately changed, "
                    f"update the _allowed set in cleo_prep.py."
                )
            if calib.get("variant") != "bansal":
                raise RuntimeError(
                    f"chr2_4s_fsnn calibration JSON variant={calib.get('variant')!r} "
                    f"does not match expected 'bansal'. Calibration is stale; re-run "
                    f"rule calibrate_opsin_plant."
                )

            opsin = UnitlessChr2FourState(
                g0=float(calib["g0"]),
                E=float(calib["E"]),
            )
            log.info(
                f"Loaded chr2_4s_fsnn calibration from {calib_path} "
                f"(hash={calib['calibration_hash']}, variant={calib['variant']}): "
                f"g0={calib['g0']:.6e}, E={calib['E']:.4f}"
            )
        else:
            raise ValueError(f"Unknown opsin_plant_class: {opsin_class!r}")
        sim.inject(opsin, fsnn.ng_main)

        # Force exponential_euler integration on Markov-based opsins (chr2_4s
        # and friends). Cleo's `Synapses(...)` call passes no `method=` arg →
        # Brian2's `'auto'` falls back to forward Euler for the time-varying-
        # coefficient 4-state Markov system. Forward Euler is unconditionally
        # unstable under bright light: with k1·Hp ~ 1.5/ms and dt=1ms,
        # dt·Ga1 > 1 produces a 2-cycle oscillation that drives C1 negative
        # on the very first integration step, fphi sign-flips after ~50
        # steps, and the whole opto-driven simulation diverges to NaN.
        # exponential_euler is exact for the linear-in-state ODE that chr2's
        # Markov is, regardless of dt, so it eliminates the artifact entirely.
        # Diagnostic in tmp/stability_sweep/diagnose_chr2_state.py confirms
        # the failure mode under default integration.
        # ProportionalCurrentOpsin has no Markov state — its synapse has no
        # state_updater — so we gate on `MarkovOpsin` to avoid AttributeError.
        from cleo.opto.opsins import MarkovOpsin
        if isinstance(opsin, MarkovOpsin):
            opsin_syn = opsin.synapses[fsnn.ng_main.name]
            opsin_syn.state_updater.method_choice = "exponential_euler"
            log.info(
                f"Set {opsin_syn.name} integrator to exponential_euler "
                "(Markov ODE is unstable under default 'euler' for bright light)"
            )

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
                sample_period = fsnn.dt * base_params.subsample_factor
                # Set up PassthroughIOProcessor with sample_period for LFP signal
                sim.set_io_processor(PassthroughIOProcessor(sample_period=sample_period))
                signals.append(lfp := RWSLFPSignalFromPSCs(name="lfp"))
                # SCLFP works quite a bit better than TKLFP from preliminary experiments.
                # signals.append(lfp := TKLFPSignal(name="lfp"))
                inj_kwargs |= dict(
                    Iampa_var_names=["vinf"],
                    Igaba_var_names=[],
                    tklfp_type="exc",
                    sample_period=sample_period,
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
