import random
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import importlib
from BRAID.BRAIDModel import BRAIDModel
from matplotlib import pyplot as plt
from snakemake.script import snakemake
from src.helpers import seed_everything
from src.scripts.cleo_prep import CleoPrep
from src.scripts.sys_id_expts import ExptData

# filepath: /home/kyle/dbgt/projects/control-of-latent-dynamics/src/scripts/ctrl_expts.py


def ctrl_expts(cond_params, base_params, cleo_prep_file, braid_model_file, outfile):
    print(f"{base_params=}")
    print(base_params.keys())
    seed = seed_everything(int(cond_params["seed"]), "ctrl_expts")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)

    k = int(cond_params["k"])
    u_max = float(base_params["u_max_mWmm2"])
    t_intertrial_ms = float(base_params.expts["t_intertrial_ms"]) // 5
    t_trial_ms = float(base_params.expts["t_trial_ms"]) // 5
    n_trials = int(base_params["n_trials_ctrl"])

    cleo_prep = CleoPrep.load(cleo_prep_file)
    cleo_prep.reset()

    # Load trained BRAID model
    braid_model: BRAIDModel = BRAIDModel.loadFromFile(braid_model_file)

    spike_counts = []
    Uopto_list = []
    Utask_list = []
    Y_list = []
    Z_list = []
    t_ms_list = []
    i_trial_boundaries = []

    # separate files so one can change without causing the other to be rerun
    setting = cond_params["setting"]
    module_name = f"src.cleo_prep_{setting}"
    module = importlib.import_module(module_name)
    light_and_probe_coords = module.light_and_probe_coords

    # Example: simple LQR-like feedback on latent state (Z)
    def compute_control(z_hat):
        # Placeholder: simple proportional controller to zero
        # Replace with your control law as needed
        u = -0.1 * z_hat
        u = np.clip(u, 0, u_max)
        return u

    class ClosedLoopProcessor(cleo.LatencyIOProcessor):
        def __init__(self):
            self.z_hat = None

        def process(self, state_dict, t_samp):
            t, i, counts = state_dict["Probe"]["mua"]
            spike_counts.append(counts)
            # Get most recent Y and Utask
            Y = cleo_prep.fsnn.state_mon.Y[:, -1]
            Utask = cleo_prep.light.irradiance_[-1]  # stand-in for task input
            # Normalize inputs
            YN = (Y - braid_model.yMean) / braid_model.yStd
            UtaskN = (Utask - braid_model.uMean) / braid_model.uStd
            # Estimate latent state
            z_hat = braid_model.encode(YN[:, None], U=UtaskN[:, None])[:, 0]
            self.z_hat = z_hat
            # Compute control
            u = compute_control(z_hat)
            Uopto_list.append(u)
            Utask_list.append(Utask)
            Y_list.append(Y)
            Z_list.append(cleo_prep.fsnn.state_mon.Z[:, -1])
            t_ms_list.append(t_samp * 1000)
            return {"light array": u * cleo_prep.light.irradiance_.units}, t_samp

    cleo_prep.sim.set_io_processor(ClosedLoopProcessor())

    for i_trial in range(n_trials):
        print(f"running closed-loop control trial {i_trial}")
        cleo_prep.run(
            t_intertrial_ms * cleo_prep.fsnn.state_mon.t.unit,
            report_period=50 * cleo_prep.fsnn.state_mon.t.unit,
        )
        i_start = len(Y_list)
        condition = cleo_prep.fsnn.sample_condition(rng)
        cleo_prep.fsnn.set_condition(condition)
        cleo_prep.run(
            t_trial_ms * cleo_prep.fsnn.state_mon.t.unit,
            report_period=50 * cleo_prep.fsnn.state_mon.t.unit,
        )
        i_end = len(Y_list)
        i_trial_boundaries.append((i_start, i_end))

    # Convert lists to arrays
    Uopto = np.array(Uopto_list)
    Utask = np.array(Utask_list)
    Y = np.array(Y_list)
    Z = np.array(Z_list)
    spikes = np.array(spike_counts)
    t_ms = np.array(t_ms_list)

    expt_data = ExptData(
        Uopto=Uopto,
        Utask=Utask,
        Y=Y,
        Z=Z,
        spikes=spikes,
        t_ms=t_ms,
        i_trial_boundaries=i_trial_boundaries,
    )
    expt_data.save(outfile)

    plt.plot(t_ms, Y)
    plt.title("Closed-loop Y over time")
    plt.savefig(outfile.replace(".pkl", "_Y.png"))


if __name__ == "__main__" and "snakemake" in locals():
    ctrl_expts(
        snakemake.wildcards,
        snakemake.params,
        snakemake.input[0],  # cleo_prep_file
        snakemake.input[1],  # braid_model_file
        snakemake.output[0],  # outfile
    )
