import brian2.only as b2
import numpy as np


def generate_trial(rng, base_params, task_params, emg):
    """returns fin,fout,T_trial,cond,i_trial_start_end,nsteps_itib,nsteps_iti"""
    C = task_params.n_conds
    cond = rng.choice(task_params.n_conds)

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
    C = emg.shape[0]
    Nout = emg.shape[2]

    nsteps_event = np.round(task_params.t_event_s * b2.second / dt).astype(int)

    nsteps_trial = np.sum(nsteps_event) + nsteps_iti
    i_trial_start_end = (nsteps_iti // 2, nsteps_trial - nsteps_iti // 2 + 1)
    assert i_trial_start_end[0] < i_trial_start_end[1] < nsteps_trial

    # Define elements of task input
    fin0 = np.zeros(nsteps_iti // 2)
    fin1 = 0.5 * np.cos((cond / C) * 2 * np.pi) * np.ones(sum(nsteps_event[:3]))
    fin2 = 0.5 * np.sin((cond / C) * 2 * np.pi) * np.ones(sum(nsteps_event[:3]))
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
