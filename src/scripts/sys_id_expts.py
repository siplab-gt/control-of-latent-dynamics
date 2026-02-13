import logging
from dataclasses import dataclass

import brian2.only as b2
import cleo
import numpy as np
from jaxtyping import Float, UInt
from matplotlib import pyplot as plt

from src.data_classes import BaseDataClass
from src.helpers import (
    preprocess_params,
    seed_everything,
    setup_debugger,
    setup_logging,
)
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
        cls, cleo_prep: CleoPrep, spike_counts, t_samp, subsample_factor, cleo_prep_file
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
        )
        assert np.all(expt_data.t_ms == expt_data.t_samp_ms)
        assert expt_data.X.shape[0] == expt_data.Y.shape[0] == expt_data.Z.shape[0], (
            f"{expt_data.X.shape}, {expt_data.Y.shape}, {expt_data.Z.shape}"
        )

        return expt_data


def run_sys_id_expts(cond_params, base_params, cleo_prep_file, outfile):
    # careful to load before setting seed, since loading sets seed as side effect
    cleo_prep = CleoPrep.load(cleo_prep_file)
    print("loaded cleo_prep before sys_id_expts", cleo_prep)
    assert cleo_prep.sim.network == cleo_prep.fsnn.net

    seed = seed_everything(int(cond_params["seed"]), "sys_id_expts")
    rng = np.random.default_rng(seed)

    k = cond_params.k
    u_max = base_params.u_max_mWmm2
    n_trials = base_params.n_trials_sys_id_max

    spike_counts = []

    class SysIDProcessor(cleo.LatencyIOProcessor):
        def process(self, state_dict, t_samp):
            if cleo_prep.probe is not None and cond_params.signals in [
                "spikes",
                "both",
            ]:
                t, i, counts = state_dict["Probe"]["mua"]
                spike_counts.append(counts)
            u_logmean = np.log(base_params.typical_Irr0_mWmm2)
            u_logstd = np.log(u_max) / 3
            value = (
                rng.lognormal(u_logmean, u_logstd, k).clip(0, u_max) * b2.mwatt / b2.mm2
            )
            return {"light array": value}, t_samp

    cleo_prep.sim.set_io_processor(
        SysIDProcessor(sample_period=cleo_prep.fsnn.dt * base_params.subsample_factor)
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
    expt_data = ExptData.from_cleo_prep(
        cleo_prep,
        spike_count_array,
        cleo_prep.sim.io_processor.t_samp,
        base_params.subsample_factor,
        cleo_prep_file,
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
