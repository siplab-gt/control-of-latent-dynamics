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
    Utask: Float[np.ndarray, "T n_task_inputs"]
    """Task-related inputs, shape (T, n_task_inputs) array"""
    Y: Float[np.ndarray, "T n"]
    """Experimental measurements, shape (T, n) array"""
    Z: Float[np.ndarray, "T n_behavior"]
    """Behavioral measurements, shape (T, n_behavior) array"""
    spikes: UInt[np.ndarray, "T m"]
    """Spike counts across channels, shape (T, m) array"""
    t_ms: Float[np.ndarray, "T"]
    """Time in ms, shape (T,) array"""
    i_trial_boundaries: UInt[np.ndarray, "n_trials 2"]
    """Trial (start, ..., end) indices, shape (n_trials, 2) array"""

    def get_trial_UYt(
        self, trial_idx: int
    ) -> tuple[
        Float[np.ndarray, "T k"], Float[np.ndarray, "T n"], Float[np.ndarray, "T"]
    ]:
        """Get the control input, experimental measurements, and time for a specific trial."""
        start, end = self.i_trial_boundaries[trial_idx]
        return self.U[start:end], self.Y[start:end], self.t_ms[start:end]

    # def save(self, filename):
    #     np.savez_compressed(
    #         filename,
    #         U=self.U,
    #         Y=self.Y,
    #         spikes=self.spikes,
    #         t_ms=self.t_ms,
    #         i_trial_boundaries=self.i_trial_boundaries,
    #     )

    # def load(cls, filename) -> "ExptData":
    #     data = np.load(filename)
    #     return cls(
    #         U=data["U"],
    #         Y=data["Y"],
    #         spikes=data["spikes"],
    #         t_ms=data["t_ms"],
    #         i_trial_boundaries=data["i_trial_boundaries"],
    #     )


def run_sys_id_expts(cond_params, base_params, cleo_prep_file, outfile):
    print(f"{base_params=}")
    print(base_params.keys())
    seed = seed_everything(int(cond_params["seed"]), "sys_id_expts")
    rng = np.random.default_rng(seed)

    k = int(cond_params["k"])
    u_max = float(base_params["u_max_mWmm2"])
    t_intertrial_ms = float(base_params.task_specific["t_intertrial_ms"]) // 5
    t_trial_ms = float(base_params.task_specific["t_trial_ms"]) // 5
    n_trials = int(cond_params["n_trials_sys_id"])

    cleo_prep = CleoPrep.load(cleo_prep_file)
    cleo_prep.reset()

    spike_counts = []

    class SysIDProcessor(cleo.LatencyIOProcessor):
        def process(self, state_dict, t_samp):
            t, i, counts = state_dict["Probe"]["mua"]
            spike_counts.append(counts)
            value = rng.uniform(0, u_max, k) * b2.mwatt / b2.mm2
            return {"light array": value}, t_samp

    cleo_prep.sim.set_io_processor(SysIDProcessor())
    # b2.prefs.codegen.target = "cython"
    conditions = []
    i_trial_boundaries = []
    for i_trial in range(12):
        print(f"running sys ID trial {i_trial}")
        cleo_prep.run(
            t_intertrial_ms * b2.ms,  # TODO
            report_period=50 * b2.second,
        )
        i_start = cleo_prep.fsnn.state_mon.Y.shape[1]
        condition = cleo_prep.fsnn.sample_condition(rng)
        cleo_prep.fsnn.set_condition(condition)
        conditions.append(condition)
        cleo_prep.run(t_trial_ms * b2.ms, report_period=50 * b2.second)
        i_end = cleo_prep.fsnn.state_mon.Y.shape[1]
        i_trial_boundaries.append((i_start, i_end))
    print("successfully ran")

    spike_count_array = np.array(spike_counts)
    assert spike_count_array.shape == (
        len(cleo_prep.fsnn.state_mon.t),
        cleo_prep.probe.signals[0].n,
    )

    # skip first sample from Cleo devices since there's a repeat after reset
    assert np.all(cleo_prep.light.t[1:] == cleo_prep.fsnn.state_mon.t)

    expt_data = ExptData(
        Uopto=cleo_prep.light.irradiance_[1:],
        # stand-in for task-related inputs for now
        Utask=cleo_prep.light.irradiance_[1:],
        Y=cleo_prep.fsnn.state_mon.Y.T[:, :10],
        Z=cleo_prep.fsnn.state_mon.Z.T[:, :10],
        spikes=spike_count_array,
        t_ms=cleo_prep.fsnn.state_mon.t / b2.ms,
        i_trial_boundaries=i_trial_boundaries,
    )
    # save dict to
    expt_data.save(outfile)

    plt.plot(cleo_prep.fsnn.v_mon.t / b2.ms, cleo_prep.fsnn.v_mon.v.T / b2.mV)
    plt.savefig(outfile.replace(".pkl", "_v.png"))


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    run_sys_id_expts(
        snakemake.wildcards, snakemake.params, snakemake.input[0], snakemake.output[0]
    )
