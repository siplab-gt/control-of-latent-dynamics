from functools import cache
import brian2.only as b2
import numpy as np

from src.helpers import load_factors_emg

DATA_FILE = __file__.replace(".py", "_data.npz")
factors, emg = load_factors_emg(DATA_FILE)
n_conds = emg.shape[0]
n_out = emg.shape[2] + 1  # output for network (2 muscles + stabilzing signal)
n_in = 3  # num inputs
g = 3  # feedback gain in spiking network
muf = -0.3  # recurrent inhibition
gs = 0.11  # gain of slow random synapses
gf = 0.13  # gain of fast random synpases
gtilde = 1.4

n_factors = None  # "P" in original code, set in init_module
Jtilde = None
V = None  #


def init_module(rng, base_params, task_params):
    """hacky init of global variables that are unique to this task"""
    global n_factors, Jtildeprime, utilde, V
    n_factors = factors.shape[1]  # "P" in original code
    n_tilde = base_params.n_rate_neurons  # TODO

    Jtilde = gtilde * 1 / np.sqrt(n_tilde) * rng.normal(size=(n_tilde, n_tilde))
    utilde = rng.uniform(-1, 1, (n_tilde, n_out))
    Jtildeprime = np.hstack([Jtilde, utilde])
    assert Jtildeprime.shape == (n_tilde, n_tilde + n_out)


def get_V(rng):
    return Jtildeprime @ V


def generate_trial(rng, base_params, task_params):
    """returns fin,fout,T_trial,cond,i_trial_start_end,nsteps_itib,nsteps_iti"""
    cond = rng.choice(n_conds)

    dt = base_params["dt_ms"] * b2.ms
    # rounding to the nearest 2dt allows for cutting in half
    nsteps_iti = (
        np.round(
            task_params["iti_min_s"]
            + rng.exponential(task_params["iti_lambda_s"]) * b2.second / (2 * dt)
        )
        * 2
    )

    # following from yijun's code
    Nout = emg.shape[2]  # just 2 muscles, not stabilizing signal

    nsteps_event = np.round(task_params.t_event_s * b2.second / dt).astype(int)

    nsteps_trial = np.sum(nsteps_event) + nsteps_iti
    i_trial_start_end = (nsteps_iti // 2, nsteps_trial - nsteps_iti // 2 + 1)
    assert i_trial_start_end[0] < i_trial_start_end[1] < nsteps_trial

    # Define elements of task input
    fin0 = np.zeros(nsteps_iti // 2)
    fin1 = 0.5 * np.cos((cond / n_conds) * 2 * np.pi) * np.ones(sum(nsteps_event[:3]))
    fin2 = 0.5 * np.sin((cond / n_conds) * 2 * np.pi) * np.ones(sum(nsteps_event[:3]))
    fin3 = np.ones(sum(nsteps_event[:1]))

    fout1 = (
        -0.0
        + 2
        * np.reshape(
            np.tile(np.reshape(np.squeeze(emg[cond, :, :]), -1), (10, 1)).T, (-1, Nout)
        ).T
    )
    fout1[0, :] = -1 * fout1[0, :]
    fout0 = -0.0 * np.ones(nsteps_iti // 2)
    fout2 = -0.5 * np.ones(nsteps_iti // 2)

    # Concatenate
    # Yijun followed the original matlab convention of NxT, but we will transpose
    fin = np.vstack(
        [
            np.hstack([fin0, fin3, np.zeros(sum(nsteps_event[1:])), fin0]),
            np.hstack([fin0, fin1, fin0]),
            np.hstack([fin0, fin2, fin0]),
        ]
    ).T

    fout = np.vstack(
        [
            np.hstack([fout0, fout1[0, :], fout0]),
            np.hstack([fout0, fout1[1, :], fout0]),
            np.hstack([fout2, -0.5 * np.ones(sum(nsteps_event)), fout2]),
        ]
    ).T

    T_trial = nsteps_trial * dt
    return fin, fout, T_trial, cond, i_trial_start_end, nsteps_iti // 2, nsteps_iti
