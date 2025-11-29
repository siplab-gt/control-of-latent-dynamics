from dataclasses import dataclass

import brian2.only as b2
import cleo
import numpy as np
from jaxtyping import Float, UInt
from matplotlib import pyplot as plt

from src.data_classes import BaseDataClass
from src.helpers import preprocess_params, seed_everything, setup_debugger
from src.log_redirector import redirect_output_to_file
from src.scripts.cleo_prep import CleoPrep
from src.scripts.plot_expt_data_trials import plot_expt_data_trials
from src.scripts.plot_expt_data_wide import plot_expt_data_wide
from src.scripts.sys_id_expts import ExptData


def impulse_responses(
    cond_params, base_params, cleo_prep_file, wide_outfile, trials_outfile
):
    # careful to load before setting seed, since loading sets seed as side effect
    cleo_prep = CleoPrep.load(cleo_prep_file)
    cleo_prep.reset()

    k = cond_params.k
    u_max = base_params.u_max_mWmm2

    spike_counts = []

    impulse_width = 20 * b2.ms
    inter_impulse_interval = 5000 * b2.ms
    t_starts = np.arange(1, k + 1) * inter_impulse_interval
    t_ends = t_starts + impulse_width

    class ImpulseResponseProcessor(cleo.LatencyIOProcessor):
        def process(self, state_dict, t_samp):
            if cleo_prep.probe is not None:
                t, i, counts = state_dict["Probe"]["mua"]
                spike_counts.append(counts)
            u = u_max * (t_samp >= t_starts) * (t_samp < t_ends) * b2.mwatt / b2.mm2
            return {"light array": u}, t_samp

    cleo_prep.sim.set_io_processor(
        ImpulseResponseProcessor(
            sample_period=cleo_prep.fsnn.dt * base_params.subsample_factor
        )
    )
    fsnn = cleo_prep.fsnn
    fsnn.record = True
    fsnn.mode = "test"
    fsnn.fin_ta = b2.TimedArray(np.zeros((1, fsnn.n_in)), fsnn.dt)
    fsnn.fout_ta = b2.TimedArray(np.zeros((1, fsnn.n_out)), fsnn.dt)
    fsnn.ytilde_ta = b2.TimedArray(np.zeros((1, fsnn.n_factors)), fsnn.dt)

    # to debug photocurrents
    opto_syn = fsnn.syn_ProportionalCurrentOpsin_ng_main
    opto_mon = b2.StateMonitor(opto_syn, "Iopto", record=True)
    assert np.all(opto_syn.Iopto == 0)

    fsnn.net.add(opto_mon)
    for i_trial in range(k + 2):
        print(f"running impulse trial {i_trial - 1}")
        cleo_prep.fsnn.net.run(
            inter_impulse_interval, namespace=fsnn._get_namespace(0 * b2.ms)
        )

    assert not np.all(opto_mon.Iopto == 0), "no photocurrents recorded!"

    print("successfully ran")

    if cleo_prep.probe is not None:
        spike_count_array = np.array(spike_counts)
        assert spike_count_array.shape == (
            len(cleo_prep.fsnn.y_mon.t),
            cleo_prep.probe.signals[0].n,
        )
    else:
        spike_count_array = np.zeros((len(cleo_prep.fsnn.y_mon.t), 0), dtype=np.uint32)

    # skip first sample from Cleo devices since there's a repeat after reset
    assert np.all(
        cleo_prep.light.t[1:] == cleo_prep.fsnn.y_mon.t[:: base_params.subsample_factor]
    )
    fsnn.i_trial_boundaries = (
        (np.column_stack([t_starts, t_starts + inter_impulse_interval]) / fsnn.dt)
        .round()
        .astype(int)
    )
    expt_data = ExptData.from_cleo_prep(
        cleo_prep,
        spike_count_array,
        cleo_prep.sim.io_processor.t_samp,
        base_params.subsample_factor,
    )
    plot_expt_data_wide(expt_data, wide_outfile)
    plot_expt_data_trials(expt_data, [trials_outfile])


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config["debug"])
    impulse_responses(
        preprocess_params(snakemake.wildcards),
        preprocess_params(snakemake.params),
        snakemake.input[0],
        snakemake.output[0],
        snakemake.output[1],
    )
