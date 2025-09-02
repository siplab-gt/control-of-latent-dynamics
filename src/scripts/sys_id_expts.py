from dataclasses import dataclass

import brian2.only as b2
import cleo
import numpy as np
from jaxtyping import Float, UInt

from src.data_classes import BaseDataClass
from src.helpers import seed_everything
from src.scripts.cleo_prep import CleoPrep


@dataclass
class ExptData(BaseDataClass):
    """should be able to use for sys ID and control experiments data"""

    U: Float[np.ndarray, "T k"]
    """Control input in mW/mm2, shape (T, k) array"""
    Y: Float[np.ndarray, "T n"]
    """Experimental measurements, shape (T, n) array"""
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
        start = self.i_trial_starts[trial_idx]
        end = self.i_trial_ends[trial_idx]
        return self.U[start:end], self.Y[start:end], self.t_ms[start:end]


def run_sys_id_expts(cond_params, base_params, cleo_prep_file, outfile):
    print(f"{base_params=}")
    print(base_params.keys())
    seed = seed_everything(int(cond_params["seed"]), "sys_id_expts")
    rng = np.random.default_rng(seed)

    k = int(cond_params["k"])
    u_max = float(base_params["u_max_mWmm2"])
    t_intertrial_ms = float(base_params.task_specific["t_intertrial_ms"]) // 10 + 0.1
    t_trial_ms = float(base_params.task_specific["t_trial_ms"]) // 10 + 0.1
    n_trials = int(cond_params["n_trials_sys_id"])

    class SysIDProcessor(cleo.LatencyIOProcessor):
        def process(self, state_dict, t_samp):
            value = rng.uniform(0, u_max, k) * b2.mwatt / b2.mm2
            return {"light array": value}, t_samp

    cleo_prep = CleoPrep.load(cleo_prep_file)
    cleo_prep.reset()
    print(b2.defaultclock.t)
    print(cleo_prep.light.t)
    cleo_prep.sim.set_io_processor(SysIDProcessor())
    # b2.prefs.codegen.target = "cython"
    conditions = []
    i_trial_boundaries = []
    for i_trial in range(n_trials):
        cleo_prep.sim.run(
            t_intertrial_ms * b2.ms,  # TODO
            report_period=50 * b2.second,
        )
        i_start = len(cleo_prep.light.irradiance_)
        condition = cleo_prep.fsnn.sample_condition(rng)
        cleo_prep.fsnn.set_condition(condition)
        conditions.append(condition)
        cleo_prep.sim.run(t_trial_ms * b2.ms, report_period=50 * b2.second)
        i_end = len(cleo_prep.light.irradiance_)
        i_trial_boundaries.append((i_start, i_end))
    print("successfully ran")

    # if using monitor, downsample to light time
    i_mon_to_keep = np.searchsorted(cleo_prep.fsnn.state_mon.t, cleo_prep.light.t)
    print(np.max(cleo_prep.fsnn.state_mon.t / b2.second))
    print(cleo_prep.light.t.max())

    expt_data = ExptData(
        U=cleo_prep.light.irradiance_,
        Y=cleo_prep.fsnn.state_mon.Y.T[i_mon_to_keep],
        t_ms=cleo_prep.fsnn.state_mon.t / b2.ms,
        i_trial_boundaries=i_trial_boundaries,
    )
    expt_data.save(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    run_sys_id_expts(
        snakemake.wildcards, snakemake.params, snakemake.input[0], snakemake.output[0]
    )
