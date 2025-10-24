import logging
import os
import shutil
import time

import brian2.only as b2
import brian2cuda  # noqa: F401
from brian2 import np

from src.fsnn import fSNN
from src.helpers import preprocess_params, setup_debugger, setup_logging

log = logging.getLogger(__name__)


def main(
    cond_params, base_params, device, fsnn_infile, fsnn_outfile, data_outfile, mode
):
    try:
        outdir = os.path.dirname(fsnn_outfile)
    except TypeError:
        outdir = os.path.dirname(data_outfile)
    save_data = data_outfile is not None
    if device == "cuda":
        log.info("deleting existing brian2cuda_output files (if any)")
        b2.set_device("cuda_standalone", build_on_run=False)
        shutil.rmtree(f"{outdir}/brian2cuda_output", ignore_errors=True)
    else:
        assert device == "cython"
        b2.prefs.codegen.target = "cython"

    if fsnn_infile is not None:
        log.info(f"loading fSNN from {fsnn_infile}")
        fsnn = fSNN.load(fsnn_infile)
    else:
        fsnn = fSNN.build(cond_params, base_params)

    n_trials = base_params.n_trials[mode].main
    n_init_trials = base_params.n_trials[mode].init
    n_settle_trials = base_params.n_trials[mode].get("settle", 0)
    log.info(
        f"Preparing {n_init_trials} warmup trials, {n_trials} {mode} trials"
        f", {n_settle_trials} settle trials"
    )
    fsnn.run_trials(
        n_trials,
        mode=mode,
        record=save_data,
        n_init_trials=n_init_trials,
        n_settle_trials=n_settle_trials,
    )
    if device == "cuda":
        log.info(f"compiling cuda code to {outdir}/brian2cuda_output")
        os.makedirs(f"{outdir}/brian2cuda_output", exist_ok=True)
        b2.device.build(
            directory=f"{outdir}/brian2cuda_output",
            # don't need to set results_directory since we aren't rerunning with
            # different run_args for each mode; rather, we are building fresh each time
            # results_directory=f"results_{mode}",
            run=False,
        )
        log.info("done building")
        log.debug(f"Scheduling summary:\n{b2.scheduling_summary(fsnn.net)}")

        start = time.time()

        b2.device.run()
        end = time.time()
        n_trials_tot = n_trials + n_init_trials + n_settle_trials
        log.info(f"done running in {end - start:.2f} seconds")
        log.info(f"{(end - start) / n_trials_tot:.2f} seconds per trial")

    log.info(f"done running {mode} trials")
    log.debug(f"{fsnn.syn_y2s.wLeft[:5, :5]=}")

    log.info(f"profiling summary:\n{b2.profiling_summary(fsnn.net)}")

    if mode == "demean":
        assert np.all(fsnn.syn_y2s.wLeft[:5, :5] == 0)
    # TODO
    # if mode == "train":
    #     assert np.sum(fsnn.syn_y2s.wLeft) != 0, (
    #         "Weights did not update during training!"
    #     )

    if fsnn_outfile is not None:
        fsnn.save(fsnn_outfile)
        log.info(f"saved fSNN to {fsnn_outfile}")

    if save_data:
        fsnn.save_mon_data(snakemake.output.data)
        log.info(f"saved fSNN {mode} data to {snakemake.output.data}")

    # device.delete() complains about deleting a bunch of stuff for brian2cuda
    # so I'll delete manually here again
    # b2.device.delete()
    if device == "cuda":
        log.info("deleting brian2cuda_output directory")
        shutil.rmtree(f"{outdir}/brian2cuda_output", ignore_errors=True)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    log_fh = setup_logging(snakemake.config, snakemake.log[0])

    fsnn = main(
        preprocess_params(snakemake.wildcards),
        preprocess_params(snakemake.params[0]),
        snakemake.config["train_device"],
        snakemake.input.get("fsnn", None),
        snakemake.output.get("fsnn", None),
        snakemake.output.get("data", None),
        mode=snakemake.wildcards.mode,
    )
