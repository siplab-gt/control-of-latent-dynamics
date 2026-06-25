import logging
from dataclasses import dataclass

import brian2.only as b2
import cleo
import numpy as np
from jaxtyping import Float, UInt
from matplotlib import pyplot as plt

from src.data_classes import BaseDataClass
from src.helpers import (
    PassthroughIOProcessor,  # Required for dill to deserialize CleoPrep with LFP
    preprocess_params,
    seed_everything,
    setup_debugger,
    setup_logging,
)
from src.opsin_monitoring import attach_opsin_monitors, extract_per_neuron
from src.scripts.cleo_prep import CleoPrep

log = logging.getLogger(__name__)


@dataclass
class ExptData(BaseDataClass):
    """should be able to use for sys ID and control experiments data"""

    Uopto: Float[np.ndarray, "T k"]
    """Control input in mW/mm2, shape (T, k) array"""
    t_opto_ms: Float[np.ndarray, "T"]
    """Time of optogenetic input samples in ms, shape (T,) array"""
    Utask: Float[np.ndarray, "T n_task_inputs"]
    """Task-related inputs, shape (T, n_task_inputs) array"""
    X: Float[np.ndarray, "T n"]
    """Ground-truth factors, shape (T, n) array"""
    Z: Float[np.ndarray, "T n_behavior"]
    """Behavioral measurements, shape (T, n_behavior) array"""
    t_ms: Float[np.ndarray, "T"]
    """Time of y_mon (recording X, Z) in ms, shape (T,) array"""
    spikes: UInt[np.ndarray, "T m"] | None
    """Spike counts across channels, shape (T, m) array. None if not recorded."""
    lfp: Float[np.ndarray, "T m"] | None
    """LFP signal across channels, shape (T, m) array, or None if not recorded"""
    t_samp_ms: Float[np.ndarray, "T_samp"]
    """Time of spike count samples in ms, shape (T_samp,) array"""
    i_trial_boundaries: UInt[np.ndarray, "n_trials 2"]
    """Trial (start, ..., end) indices, shape (n_trials, 2) array"""
    conditions: list[int]
    """List of condition indices for each trial"""
    cleo_prep_file: str | None
    Iopto_per_neuron: Float[np.ndarray, "T N"] | None = None
    """Per-neuron biophysical photocurrent from opsin plant, shape (T, N) array.
    Subsampled on the same grid as X/Y/Z. Used by USID's opsin-fit diagnostic
    to compare the NN opsin's light→photocurrent mapping against ground truth.
    None if the plant opsin didn't expose a readable Iopto variable."""
    channel_neuron_transmittance: Float[np.ndarray, "k N"] | None = None
    """Per-channel→per-neuron transmittance matrix, shape (k_opto, N). Static
    (channel geometry doesn't change during a trial). Multiply by Uopto(t) to
    reconstruct per-neuron irradiance. None if Iopto_per_neuron is None."""
    V_per_neuron: Float[np.ndarray, "T N"] | None = None
    """Per-neuron membrane voltage from the plant, shape (T, N) array, on the
    same grid as Iopto_per_neuron. Enables decomposing realized opsin saturation
    into the light-gating vs the voltage-dependent (V−E) driving-force factors.
    Recorded by ctrl_expts (record_v=True); None for sys-ID and natural-electrode
    data (record_v=False) and whenever Iopto_per_neuron is None."""

    @property
    def n_trials(self) -> int:
        return self.i_trial_boundaries.shape[0]

    def get_trial_UXt(
        self, trial_idx: int
    ) -> tuple[
        Float[np.ndarray, "T k"], Float[np.ndarray, "T n"], Float[np.ndarray, "T"]
    ]:
        """Get the control input, experimental measurements, and time for a specific trial."""
        start, end = self.i_trial_boundaries[trial_idx]
        t_start = self.t_ms[start]
        t_end = self.t_ms[end] if end < len(self.t_ms) else self.t_ms[-1] + 1
        start_end_opto = slice(*np.searchsorted(self.t_opto_ms, (t_start, t_end)))
        return self.Uopto[start_end_opto], self.X[start:end], self.t_ms[start:end]

    @property
    def Y(self) -> Float[np.ndarray, "T 2*m"]:
        if self.spikes is not None and self.lfp is not None:
            Y = np.hstack([self.spikes, self.lfp])
        elif self.spikes is not None:
            Y = self.spikes
        elif self.lfp is not None:
            Y = self.lfp
        else:
            Y = np.empty((self.X.shape[0], 0))
        return Y

    @classmethod
    def from_cleo_prep(
        cls, cleo_prep: CleoPrep, spike_counts, t_samp, subsample_factor, cleo_prep_file,
        Iopto_per_neuron=None, channel_neuron_transmittance=None, V_per_neuron=None,
    ):
        i_trial_bounds = np.array(cleo_prep.fsnn.i_trial_boundaries) / subsample_factor
        i_trial_bounds[:, 0] = i_trial_bounds[:, 0].astype(int)
        i_trial_bounds[:, 1] = np.ceil(i_trial_bounds[:, 1]).astype(int)
        i_trial_bounds = i_trial_bounds.astype(np.uint32)

        if cleo_prep.lfp is not None:
            lfp = cleo_prep.lfp.lfp
            if isinstance(cleo_prep.lfp, cleo.ephys.TKLFPSignal):
                lfp /= b2.uvolt
        else:
            lfp = None

        expt_data = cls(
            Uopto=cleo_prep.light.irradiance_[1:],
            t_opto_ms=cleo_prep.light.t[1:] / b2.ms,
            # stand-in for task-related inputs for now
            Utask=cleo_prep.fsnn.in_mon.fin[:, ::subsample_factor].T,
            X=cleo_prep.fsnn.y_mon.y_[:, ::subsample_factor].T,
            Z=cleo_prep.fsnn.z_mon.z[:, ::subsample_factor].T,
            t_ms=cleo_prep.fsnn.y_mon.t[::subsample_factor] / b2.ms,
            spikes=spike_counts,
            lfp=lfp,
            t_samp_ms=t_samp / b2.ms,
            i_trial_boundaries=i_trial_bounds,
            conditions=cleo_prep.fsnn.conditions,
            cleo_prep_file=cleo_prep_file,
            Iopto_per_neuron=Iopto_per_neuron,
            channel_neuron_transmittance=channel_neuron_transmittance,
            V_per_neuron=V_per_neuron,
        )
        assert np.all(expt_data.t_ms == expt_data.t_samp_ms)
        assert expt_data.X.shape[0] == expt_data.Y.shape[0] == expt_data.Z.shape[0], (
            f"{expt_data.X.shape}, {expt_data.Y.shape}, {expt_data.Z.shape}"
        )
        if expt_data.Iopto_per_neuron is not None:
            assert expt_data.Iopto_per_neuron.shape[0] == expt_data.X.shape[0], (
                f"Iopto_per_neuron T={expt_data.Iopto_per_neuron.shape[0]} "
                f"!= X T={expt_data.X.shape[0]}"
            )
            assert expt_data.channel_neuron_transmittance is not None, (
                "channel_neuron_transmittance required when Iopto_per_neuron is provided"
            )
            assert expt_data.channel_neuron_transmittance.shape[1] == expt_data.Iopto_per_neuron.shape[1], (
                f"transmittance N={expt_data.channel_neuron_transmittance.shape[1]} "
                f"!= Iopto N={expt_data.Iopto_per_neuron.shape[1]}"
            )
        if expt_data.V_per_neuron is not None:
            assert expt_data.V_per_neuron.shape == expt_data.Iopto_per_neuron.shape, (
                f"V_per_neuron {expt_data.V_per_neuron.shape} "
                f"!= Iopto_per_neuron {expt_data.Iopto_per_neuron.shape}"
            )

        return expt_data


def run_sys_id_expts(cond_params, base_params, cleo_prep_file, outfile):
    # careful to load before setting seed, since loading sets seed as side effect
    cleo_prep = CleoPrep.load(cleo_prep_file)
    print("loaded cleo_prep before sys_id_expts", cleo_prep)
    assert cleo_prep.sim.network == cleo_prep.fsnn.net

    natural_only = getattr(base_params, 'natural_only', False)
    salt = "natural_electrode_expts" if natural_only else "sys_id_expts"
    seed = seed_everything(int(cond_params["seed"]), salt)
    rng = np.random.default_rng(seed)

    k = cond_params.k
    if natural_only:
        n_trials = int(base_params.n_trials_natural)
    else:
        n_trials = int(base_params.n_trials_sys_id_max)

    spike_counts = []

    class SysIDProcessor(cleo.LatencyIOProcessor):
        def process(self, state_dict, t_samp):
            if cleo_prep.probe is not None and cond_params.signals in [
                "spikes",
                "both",
            ]:
                t, i, counts = state_dict["Probe"]["mua"]
                spike_counts.append(counts)
            if natural_only:
                # Zero irradiance → zero photocurrent → pure natural dynamics
                value = np.zeros(k) * b2.mwatt / b2.mm2
            else:
                # Per-channel lognormal opto drive, parameterized by two
                # sys_id_expts wildcards (there is no named "regime" — the regime
                # is a point in this 2-D knob space):
                #   stim_u_max_mWmm2 — amplitude clip AND lognormal spread
                #     (u_logstd = log(cap)/3). Sub-saturating when below the opsin
                #     half-saturation (i_half≈49 mW/mm²) → clean linear B'/Cz2 fit.
                #   stim_active_frac — per-channel per-step Bernoulli ON-probability
                #     (sparsity); <1 decorrelates channels so the sys-ID covariance
                #     resolves individual opto→Z directions.
                # Median is typical_Irr0. Defaults (30, 0.5) give the control-regime
                # excitation the validated USID+opsin runs use; (75, 1.0) reproduces
                # the legacy dense saturating drive.
                stim_u_max = float(cond_params.stim_u_max_mWmm2)
                p_active = float(cond_params.stim_active_frac)
                u_logmean = np.log(base_params.typical_Irr0_mWmm2)
                u_logstd = np.log(stim_u_max) / 3
                amp = rng.lognormal(u_logmean, u_logstd, k).clip(0, stim_u_max)
                amp = amp * (rng.random(k) < p_active)
                value = amp * b2.mwatt / b2.mm2
            return {"light array": value}, t_samp

    cleo_prep.sim.set_io_processor(
        SysIDProcessor(sample_period=cleo_prep.fsnn.dt * base_params.subsample_factor)
    )

    # Record per-neuron photocurrent (Iopto) for opsin-fit validation, shared with
    # ctrl_expts via src.opsin_monitoring. record_v=False -> no membrane voltage
    # (V_per_neuron stays None), which is all sys-ID needs. record_dt records Iopto
    # directly on the X/Y/Z subsample grid, so the StateMonitor holds ~subsample_factor×
    # fewer samples (the full-resolution array is ~681 MB and bloats the saved
    # sys_id_data / natural_electrode .pkl.gz); the sampled values match X/Y/Z.
    record_dt = base_params.subsample_factor * cleo_prep.fsnn.dt
    monitors, channel_neuron_transmittance = attach_opsin_monitors(
        cleo_prep, record_v=False, record_dt=record_dt
    )

    log.info(f"running {n_trials} sys ID trials")
    cleo_prep.fsnn.run_trials(
        n_trials,
        mode="test",
        rng=rng,
        record=True,
        n_init_trials=0,
        n_settle_trials=0,
    )
    print("successfully ran")

    if cleo_prep.probe is not None and cond_params.signals in ["spikes", "both"]:
        spike_count_array = np.array(spike_counts)
        assert spike_count_array.shape == (
            len(cleo_prep.sim.io_processor.t_samp),
            cleo_prep.probe.signals[0].n,
        )
    else:
        spike_count_array = None
    # skip first sample from Cleo devices since there's a repeat after reset
    assert np.all(
        cleo_prep.light.t[1:] == cleo_prep.fsnn.y_mon.t[:: base_params.subsample_factor]
    )

    # Already recorded on the X/Y/Z subsample grid via record_dt -> no further subsample.
    Iopto_per_neuron, _ = extract_per_neuron(monitors, 1)

    expt_data = ExptData.from_cleo_prep(
        cleo_prep,
        spike_count_array,
        cleo_prep.sim.io_processor.t_samp,
        base_params.subsample_factor,
        cleo_prep_file,
        Iopto_per_neuron=Iopto_per_neuron,
        channel_neuron_transmittance=channel_neuron_transmittance,
    )
    expt_data.save(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    log_fh = setup_logging(snakemake.config, snakemake.log[0])
    setup_debugger(snakemake.config["debug"])

    run_sys_id_expts(
        preprocess_params(snakemake.wildcards),
        preprocess_params(snakemake.params),
        snakemake.input[0],
        snakemake.output[0],
    )
