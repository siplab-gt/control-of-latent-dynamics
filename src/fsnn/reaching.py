from dataclasses import dataclass, field

import brian2.only as b2
import numba
import numpy as np

from src.fsnn.task import fSNNTask
from src.helpers import AttrDict, load_factors_emg, orth, setup_debugger


@numba.njit(cache=True)
def _rate_model(fin, fout, nsteps_trial, x, Jtilde, utilde, utilde_in, etau, n_tilde):
    """returns (xprimes, x)"""
    xprimes = np.zeros((nsteps_trial, n_tilde))

    F = np.tanh
    for i in range(nsteps_trial):
        xinf = F(Jtilde @ x + utilde @ fout[i] + utilde_in @ fin[i])
        x = xinf + (x - xinf) * etau
        xprimes[i] = x

    return xprimes, x


@dataclass
class ReachingTask(fSNNTask):
    emg: np.ndarray = field(default=None, repr=False)
    """(n_conds, n_timepoints, n_muscles)"""
    Jtilde: np.ndarray = field(default=None, repr=False)
    """(n_tilde, n_tilde), recurrent weights of rate network"""
    utilde: np.ndarray = field(default=None, repr=False)
    """(n_tilde, n_out), maps from outputs to rate network"""
    utilde_in: np.ndarray = field(default=None, repr=False)
    """(n_tilde, n_in), maps from inputs to rate network"""
    U: np.ndarray = field(default=None, repr=False)
    """(n_ng, n_tilde) orthogonal matrix of norm g that maps from space of Ntilde to N"""
    V: np.ndarray = field(default=None, repr=False)
    """(n_tilde, n_factors)"""
    task_params: AttrDict = None
    """task parameters"""

    # defaults for task, would be overridden in multitask setting
    g: float = 3  # feedback gain in spiking network
    muf: float = -0.3  # recurrent inhibition
    gs: float = 0.11  # gain of slow random synapses
    gf: float = 0.13  # gain of fast random synpases
    gtilde: float = 1.4  # feedback gain in rate network
    n_in: int = 3  # num inputs

    F = np.tanh
    tau = 10 * b2.ms

    @property
    def etau(self):
        return np.exp(-self.dt / self.tau)

    @property
    def n_out(self):
        return (
            self.emg.shape[2] + 1
        )  # output for network (2 muscles + stabilzing signal)

    @property
    def n_conds(self):
        return self.emg.shape[0]

    @property
    def n_tilde(self):
        return len(self.Jtilde)

    @property
    def n_factors(self):
        """ "P" in original code, set in init_module"""
        return self.u.shape[1]

    def _trial_reaching(self, rng):
        """follows trial_reaching.m
        returns fin,fout,nsteps_trial,cond,i_trial_start_end,nsteps_iti"""
        cond = rng.choice(self.n_conds)

        # rounding to the nearest 2dt allows for cutting in half
        nsteps_iti = (
            np.round(
                (
                    self.task_params.iti_min
                    + rng.exponential(self.task_params.iti_lambda / b2.second)
                    * b2.second
                )
                / (2 * self.dt)
            )
            * 2
        )

        # following from yijun's code
        nsteps_event = np.round(self.task_params.t_event / self.dt).astype(int)

        nsteps_trial = int(np.sum(nsteps_event) + nsteps_iti)
        nsteps_iti_half = int(nsteps_iti // 2)
        i_trial_start_end = (nsteps_iti_half, nsteps_trial - nsteps_iti_half + 1)
        assert i_trial_start_end[0] < i_trial_start_end[1] < nsteps_trial

        # Define elements of task input
        fin0 = np.zeros(nsteps_iti_half)
        fin1 = (
            0.5
            * np.cos((cond / self.n_conds) * 2 * np.pi)
            * np.ones(sum(nsteps_event[:3]))
        )
        fin2 = (
            0.5
            * np.sin((cond / self.n_conds) * 2 * np.pi)
            * np.ones(sum(nsteps_event[:3]))
        )
        fin3 = np.ones(sum(nsteps_event[:1]))

        fout1 = 2 * np.repeat(self.emg[cond], 10, axis=0).T
        assert fout1.shape[1] == nsteps_trial - nsteps_iti == sum(nsteps_event), (
            f"{fout1.shape=}"
        )
        fout1[0, :] = -1 * fout1[0, :]
        fout0 = -0.0 * np.ones(nsteps_iti_half)
        fout2 = -0.5 * np.ones(nsteps_iti_half)
        # 3rd output is just -0.5 the whole time

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
                # 3rd output is just -0.5 the whole time, but will keep this for comparison with legacy
                np.hstack([fout2, -0.5 * np.ones(sum(nsteps_event[:3])), fout2]),
            ]
        ).T

        T_trial = nsteps_trial * self.dt

        i_start, i_end = i_trial_start_end
        assert i_start < i_end < nsteps_trial
        return fin, fout, nsteps_trial, cond, i_trial_start_end, nsteps_iti

    def _rate_model(self, fin, fout, nsteps_trial, x=None, rng=None):
        """returns (xprimes, x)"""
        if x is None:
            x = rng.normal(size=self.n_tilde)

        assert len(fin) == len(fout) == nsteps_trial

        return _rate_model(
            fin,
            fout,
            nsteps_trial,
            x,
            self.Jtilde,
            self.utilde,
            self.utilde_in,
            self.etau,
            self.n_tilde,
        )

    def generate_trial(
        self, rng, ctrl=False, x=None, W=None, **kwargs
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, int, int, tuple[int, int], int, dict
    ]:
        """returns ytilde, fout, fin, nsteps_trial, cond, i_trial_start_end, nsteps_iti, gen_trial_kwargs"""
        # Compute target output for rate network
        fin, fout, nsteps_trial, cond, i_trial_start_end, nsteps_iti = (
            self._trial_reaching(rng)
        )
        xprimes, x = self._rate_model(fin, fout, nsteps_trial, x, rng=rng)
        ytilde = xprimes @ self.V

        # for control experiments, remove direction-specifying input
        # and make fout (Zref) what is actually attainable from factors
        if ctrl:
            if W is None or not isinstance(W, np.ndarray):
                raise ValueError(
                    "When ctrl=True, W must be provided as a numpy.ndarray of shape (n_out, n_factors + 1)."
                )
            assert W.shape == (self.n_out, self.n_factors + 1), W.shape
            fin[:, 1:] = 0
            yplus = np.concatenate((ytilde, np.ones((len(ytilde), 1))), axis=1)
            fout = yplus @ W.T
            assert fout.shape == (len(ytilde), self.n_out)

        return (
            ytilde,
            fout,
            fin,
            nsteps_trial,
            cond,
            i_trial_start_end,
            nsteps_iti,
            {"x": x},
        )


def init_task(seed, base_params, task_params, n_ng, g, n_latent_dims_offset):
    rng = np.random.default_rng(seed + 19320812)
    clss = ReachingTask
    DATA_FILE = __file__.replace(".py", "_data.npz")
    _, emg = load_factors_emg(DATA_FILE)
    n_tilde = task_params.n_rate_neurons
    dt = base_params.dt
    U = g * orth(rng.uniform(-1, 1, (n_ng, n_tilde)))
    utilde_in = rng.uniform(-1, 1, (n_tilde, clss.n_in))
    u_in = U @ utilde_in

    n_out = emg.shape[2] + 1
    Jtilde = clss.gtilde / np.sqrt(n_tilde) * rng.normal(size=(n_tilde, n_tilde))
    utilde = rng.uniform(-1, 1, (n_tilde, n_out))
    # NOTE: don't know why Jtildeprime is used in multitask but not reaching demo
    # Jtildeprime = np.hstack([Jtilde, utilde])

    reaching_task = clss(
        u_in=u_in,
        dt=dt,
        emg=emg,
        Jtilde=Jtilde,
        utilde=utilde,
        utilde_in=utilde_in,
        U=U,
        V=None,  # will get from PCA of rate model
        u=None,  # need V to compute
        task_params=task_params,
    )

    XX = np.zeros((n_tilde, n_tilde))

    for i in range(task_params.n_trials_pca):
        fin, fout, nsteps_trial, cond, i_trial_start_end, nsteps_iti = (
            reaching_task._trial_reaching(rng)
        )
        if i == 0:
            xprimes, x = reaching_task._rate_model(fin, fout, nsteps_trial, rng=rng)
        else:
            xprimes, x = reaching_task._rate_model(fin, fout, nsteps_trial, x)
        XX += xprimes.T @ xprimes
        print(f"PCA, {i + 1} trials of {task_params.n_trials_pca}")
        d, V = np.linalg.eigh(XX)
        # print(f"  top 5 eigenvalues: {d[-1:-6:-1]}")
        print(
            f"  pct variance explained by top 5 PCs: {np.cumsum(d)[-1:-6:-1] / np.sum(d)}"
        )
        print()

    # Eigen decomposition and PCA
    d, V = np.linalg.eigh(XX)
    V = np.fliplr(V)
    d = np.flipud(d)

    fracPCA = 0.99
    P = np.argmax(np.cumsum(d) / np.sum(d) > fracPCA) + 1 + n_latent_dims_offset

    V = V[:, :P]
    reaching_task.V = V
    reaching_task.u = U @ Jtilde @ V

    return reaching_task
