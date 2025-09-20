import importlib
import random
from dataclasses import dataclass

import brian2.only as b2
import cleo
import numpy as np
import tensorflow as tf
from BRAID.BRAIDModel import BRAIDModel
from matplotlib import pyplot as plt
from snakemake.script import snakemake

from src.ctrl import Controller
from src.helpers import seed_everything
from src.scripts.cleo_prep import CleoPrep
from src.scripts.sys_id_expts import ExptData


def ctrl_expts(
    cond_params, base_params, cleo_prep_file, braid_model_file, outfile, config
):
    print(f"{base_params=}")
    print(base_params.keys())
    seed = seed_everything(int(cond_params["seed"]), "ctrl_expts")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)

    k = int(cond_params["k"])
    u_max = float(base_params["u_max_mWmm2"])
    t_intertrial = float(base_params.expts["t_intertrial_ms"]) * b2.ms
    t_trial = float(base_params.expts["t_trial_ms"]) * b2.ms
    n_trials = int(base_params["n_trials_ctrl"])
    report_period = config["report_period_s"] * b2.second

    cleo_prep = CleoPrep.load(cleo_prep_file)
    cleo_prep.reset()

    # Load trained BRAID model
    braid_model: BRAIDModel = BRAIDModel.loadFromFile(braid_model_file)

    # separate files so one can change without causing the other to be rerun
    ctrl_type = cond_params["ctrl_type"]
    ControllerType = getattr(
        importlib.import_module(f"src.ctrl.{ctrl_type}"),
        f"{ctrl_type.upper()}Controller",
    )
    ctrl_proc: Controller = ControllerType(
        k_opto=k,
        k_task=cleo_prep.fsnn.k_task,
        m=braid_model.ny,
        rng=rng,
        u_max_mWmm2=u_max,
        braid_model=braid_model,
        baseline_latency=float(base_params["baseline_latency_ms"]) * b2.ms,
        sample_period=float(base_params["dt_ms"]) * b2.ms,
        u_task=cleo_prep.fsnn.u_task_,
    )

    ctrl_proc.compute_latency = ctrl_proc.measure_latency()
    print(f"measured control computation latency: {ctrl_proc.compute_latency}")

    cleo_prep.sim.set_io_processor(ctrl_proc)

    i_trial_boundaries = []
    for i_trial in range(n_trials):
        print(f"running control trial {i_trial}")
        cleo_prep.run(t_intertrial, report_period=report_period)
        i_start = cleo_prep.fsnn.state_mon.Y.shape[1]
        condition = cleo_prep.fsnn.sample_condition(rng)
        cleo_prep.fsnn.set_condition(condition)
        cleo_prep.run(t_trial, report_period=report_period)
        i_end = cleo_prep.fsnn.state_mon.Y.shape[1]
        i_trial_boundaries.append((i_start, i_end))

    expt_data = ExptData(
        Uopto=cleo_prep.light.irradiance_[1:],
        t_opto_ms=cleo_prep.light.t[1:] / b2.ms,
        # stand-in for task-related inputs for now
        Utask=cleo_prep.light.irradiance_[1:],
        Y=cleo_prep.fsnn.state_mon.Y.T[:, :10],
        Z=cleo_prep.fsnn.state_mon.Z.T[:, :10],
        t_ms=cleo_prep.fsnn.state_mon.t / b2.ms,
        spikes=np.array(ctrl_proc.spike_counts),
        t_samp_ms=ctrl_proc.t_samp / b2.ms,
        i_trial_boundaries=np.array(i_trial_boundaries),
    )
    expt_data.save(outfile)

    expt_data.plot_all(outfile.replace(".pkl", "_plots.png"))


if __name__ == "__main__" and "snakemake" in locals():
    ctrl_expts(
        snakemake.wildcards,
        snakemake.params,
        snakemake.input.cleo_prep[0],
        snakemake.input.braid_model[0],
        snakemake.output[0],  # outfile
        snakemake.config,  # for non-essential params
    )
