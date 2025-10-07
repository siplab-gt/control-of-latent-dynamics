from dataclasses import dataclass

import brian2.only as b2
import cleo
import numpy as np
from jaxtyping import Float, UInt
from matplotlib import pyplot as plt

from src.data_classes import BaseDataClass
from src.helpers import seed_everything
from src.scripts.cleo_prep import CleoPrep


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
    """Time of state_mon (recording X, Z) in ms, shape (T,) array"""
    spikes: UInt[np.ndarray, "T m"]
    """Spike counts across channels, shape (T, m) array"""
    t_samp_ms: Float[np.ndarray, "T_samp"]
    """Time of spike count samples in ms, shape (T_samp,) array"""
    i_trial_boundaries: UInt[np.ndarray, "n_trials 2"]
    """Trial (start, ..., end) indices, shape (n_trials, 2) array"""

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

    @classmethod
    def from_cleo_prep(
        cls, cleo_prep: CleoPrep, spike_counts, t_samp, i_trial_boundaries
    ):
        expt_data = cls(
            Uopto=cleo_prep.light.irradiance_[1:],
            t_opto_ms=cleo_prep.light.t[1:] / b2.ms,
            # stand-in for task-related inputs for now
            Utask=cleo_prep.fsnn.u_task(t_samp),
            X=cleo_prep.fsnn.state_mon.X.T[:, : cleo_prep.fsnn.n_pcs],
            Z=cleo_prep.fsnn.state_mon.Z.T[:, :4],
            t_ms=cleo_prep.fsnn.state_mon.t / b2.ms,
            spikes=spike_counts,
            t_samp_ms=t_samp / b2.ms,
            i_trial_boundaries=np.array(i_trial_boundaries),
        )
        assert np.all(expt_data.t_ms == expt_data.t_samp_ms)
        return expt_data

    # def save(self, filename):
    #     np.savez_compressed(
    #         filename,
    #         U=self.U,
    #         X=self.X,
    #         spikes=self.spikes,
    #         t_ms=self.t_ms,
    #         i_trial_boundaries=self.i_trial_boundaries,
    #     )

    # def load(cls, filename) -> "ExptData":
    #     data = np.load(filename)
    #     return cls(
    #         U=data["U"],
    #         X=data["X"],
    #         spikes=data["spikes"],
    #         t_ms=data["t_ms"],
    #         i_trial_boundaries=data["i_trial_boundaries"],
    #     )


def run_sys_id_expts(cond_params, base_params, cleo_prep_file, outfile):
    print(f"{base_params=}")
    print(base_params.keys())

    # careful to load before setting seed, since loading sets seed as side effect
    cleo_prep = CleoPrep.load(cleo_prep_file)
    cleo_prep.reset()
    print("loaded cleo_prep before sys_id_expts", cleo_prep)
    assert cleo_prep.sim.network == cleo_prep.fsnn.net

    seed = seed_everything(int(cond_params["seed"]), "sys_id_expts")
    rng = np.random.default_rng(seed)

    k = int(cond_params["k"])
    u_max = float(base_params["u_max_mWmm2"])
    t_intertrial_ms = float(base_params.expts["t_intertrial_ms"])
    t_trial_ms = float(base_params.expts["t_trial_ms"])
    n_trials = int(base_params["n_trials_sys_id_max"])

    spike_counts = []

    class SysIDProcessor(cleo.LatencyIOProcessor):
        def process(self, state_dict, t_samp):
            if cleo_prep.probe is not None:
                t, i, counts = state_dict["Probe"]["mua"]
                spike_counts.append(counts)
            value = rng.uniform(0, u_max, k) * b2.mwatt / b2.mm2
            return {"light array": value}, t_samp

    cleo_prep.sim.set_io_processor(
        SysIDProcessor(sample_period=base_params.sample_period_ms * b2.ms)
    )
    # b2.prefs.codegen.target = "cython"
    conditions = []
    i_trial_boundaries = []
    for i_trial in range(n_trials):
        print(f"running sys ID trial {i_trial}")
        # TODO: likely will sample trials at the very beginning, not on the go.
        # needed for standalone mode
        cleo_prep.run(
            t_intertrial_ms * b2.ms,  # TODO
            report_period=50 * b2.second,
        )
        i_start = cleo_prep.fsnn.state_mon.X.shape[1]
        condition = cleo_prep.fsnn.sample_condition(rng)
        cleo_prep.fsnn.set_condition(condition)
        conditions.append(condition)
        cleo_prep.run(t_trial_ms * b2.ms, report_period=50 * b2.second)
        i_end = cleo_prep.fsnn.state_mon.X.shape[1]
        i_trial_boundaries.append((i_start, i_end))
    print("successfully ran")

    if cleo_prep.probe is not None:
        spike_count_array = np.array(spike_counts)
        assert spike_count_array.shape == (
            len(cleo_prep.fsnn.state_mon.t),
            cleo_prep.probe.signals[0].n,
        )
    else:
        spike_count_array = np.zeros(
            (len(cleo_prep.fsnn.state_mon.t), 0), dtype=np.uint32
        )

    # skip first sample from Cleo devices since there's a repeat after reset
    assert np.all(cleo_prep.light.t[1:] == cleo_prep.fsnn.state_mon.t)

    expt_data = ExptData.from_cleo_prep(
        cleo_prep,
        spike_count_array,
        cleo_prep.sim.io_processor.t_samp,
        i_trial_boundaries,
    )
    expt_data.save(outfile)

    plt.plot(cleo_prep.fsnn.v_mon.t / b2.ms, cleo_prep.fsnn.v_mon.v.T[:, :10] / b2.mV)
    plt.savefig(outfile.replace(".pkl.gz", "_v.png"))


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    run_sys_id_expts(
        snakemake.wildcards, snakemake.params, snakemake.input[0], snakemake.output[0]
    )
