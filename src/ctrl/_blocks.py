"""Closed-loop block assembly shared by the controller rollout and the offline metrics.

`NLMPC` iterates `x_{t+1} = A_cl x_t + B_block u_t` inside the solver, and the attribution
metrics (`src/scripts/attribution_metrics.py`) replay the same recursion offline to form the
light-off counterfactual. Both build the blocks here so a change to the propagation cannot
diverge between what the controller planned and what the metrics score.
"""
import numpy as np


def is_subspace(lds) -> bool:
    """Whether the LDS carries a natural/unnatural block decomposition (USID) or is unified (BRAID)."""
    return hasattr(lds, 'A11')


def closed_loop_blocks(lds):
    """Assemble `(A_cl, B_block)` for a subspace LDS, as numpy.

    Returns:
        A_cl: (n, n) closed-loop state matrix, `n = n1 + n2`.
        B_block: (n, nu) stacked `[B1'; B2']`.

    The observer-form update

        x1' = A11 x1 + K1 (C1 x1 + C2 x2) + B1' u + A12 x2
        x2' = A22 x2 + K2 (C2 x2)         + B2' u

    is exactly linear in `x = [x1; x2]`, because the innovation `y_pred − C1 x1` equals `C2 x2`.
    So it collapses to a single block system with

        A_cl = [[A11 + K1 C1,  K1 C2 + A12], [0,  A22 + K2 C2]].

    The lower-left block is zero: the natural block is not driven by the unnatural innovation.
    """
    n1, n2 = lds.A11.shape[0], lds.A22.shape[0]
    A_cl = np.zeros((n1 + n2, n1 + n2), dtype=float)
    A_cl[:n1, :n1] = lds.A11 + lds.K1 @ lds.C1
    top_right = lds.K1 @ lds.C2
    if lds.A12 is not None:
        top_right = top_right + lds.A12
    A_cl[:n1, n1:] = top_right
    A_cl[n1:, n1:] = lds.A22 + lds.K2 @ lds.C2
    B_block = np.concatenate([lds.B1_prime, lds.B2_prime], axis=0).astype(float)
    return A_cl, B_block


def unified_blocks(lds):
    """Assemble `(A, B_opto)` for a unified LDS, as numpy.

    BRAID's planning rollout is a deterministic forward sim with no observer correction — its LDS
    exposes no gain in the K1/K2 form the subspace observer folds into `A_cl`, and the Kalman
    update happens out of band in `Controller.process`. So the free-run matrix is `A` itself.
    """
    return np.asarray(lds.A, dtype=float), np.asarray(lds.B_opto, dtype=float)


def readout_blocks(lds):
    """Behavior-readout blocks `(Cz1, Cz2, d_z)` of a subspace LDS.

    Returns:
        Cz1: (nz, n1), Cz2: (nz, n2), d_z: (nz,) — the affine bias on the behavior rows.

    Raises if the LDS does not carry the split, since every quantity built on it (the α objective,
    `rho`, `pi`, the P2 counterfactual) is undefined without one.
    """
    if lds.Cz1 is None or lds.Cz2 is None:
        raise AttributeError(
            "subspace LDS carries no Cz1/Cz2 split. The attribution objective and metrics "
            "require the behavior readout to be separable into natural and induced blocks"
        )
    nz = lds.Cz1.shape[0]
    d_z = np.asarray(lds.d, dtype=float)[-nz:]
    return (
        np.asarray(lds.Cz1, dtype=float),
        np.asarray(lds.Cz2, dtype=float),
        d_z,
    )


def lfp_readout_blocks(lds):
    """Neural-readout blocks `(Cy1, Cy2)` — the LFP rows of the augmented `C1`/`C2`.

    The augmented readout stacks `[LFP | behavior]`, so the LFP rows are everything above the
    `nz` behavior rows. Routing is measured here rather than on behavior because both blocks land
    in the same units on the recorded readout.
    """
    nz = lds.Cz1.shape[0] if lds.Cz1 is not None else 0
    ny_lfp = lds.C1.shape[0] - nz
    return (
        np.asarray(lds.C1, dtype=float)[:ny_lfp],
        np.asarray(lds.C2, dtype=float)[:ny_lfp],
    )


def task_input_block(lds):
    """`B_task` — the state coupling of the modelled task input — or None when there is none.

    `LDS.B_task` is `B[:, k_opto:]`, empty whenever the fit ran with `model_task_inputs=False`.
    USID's `model_fit` currently refuses `model_task_inputs=True` outright, so this returns None
    for every USID fit until that coupling is implemented.
    """
    B_task = getattr(lds, "B_task", None)
    if B_task is None or np.asarray(B_task).shape[1] == 0:
        return None
    return np.asarray(B_task, dtype=float)


def observer_step(lds, x1, x2, y_zscored, u_opto_zscored):
    """Single subspace observer step in predictor form.

    Args:
        lds: subspace LDS with A11, A22, A12, K1, K2, C1, B1_prime, B2_prime, d.
        x1: (n1,) natural state at time t.
        x2: (n2,) unnatural state at time t.
        y_zscored: (m,) z-scored augmented measurement [LFP | behavior] at time t.
        u_opto_zscored: (k_opto,) z-scored opto input at time t.

    Returns:
        (x1_next, x2_next) at time t+1.

    `lds.d` is the affine output bias (ỹ = C·x + d); subtracting it leaves the innovation and the
    K1 drive acting on the C·x residual the observer gains were fitted against.
    """
    innovation = (y_zscored - lds.d) - lds.C1 @ x1
    x1_next = lds.A11 @ x1 + lds.K1 @ (y_zscored - lds.d) + lds.B1_prime @ u_opto_zscored
    if lds.A12 is not None:
        x1_next += lds.A12 @ x2
    x2_next = lds.A22 @ x2 + lds.K2 @ innovation + lds.B2_prime @ u_opto_zscored
    return x1_next, x2_next


def observer_states(lds, y_zscored, u_opto_zscored, x0=None):
    """Replay the observer over a recorded session, returning its state at every sample.

    Args:
        lds: subspace LDS.
        y_zscored: (T, m) z-scored augmented measurements.
        u_opto_zscored: (T, k_opto) z-scored opto inputs.
        x0: (n,) initial state, or None to start from zero as the deployed controller does.

    Returns:
        (T, n) states `x_t`, where `x_t` was formed from measurements up to `t-1` — the same
        predictor-form convention the controller logs into `Xhat_lds`.
    """
    n1, n2 = lds.A11.shape[0], lds.A22.shape[0]
    T = y_zscored.shape[0]
    out = np.zeros((T, n1 + n2), dtype=float)
    if x0 is not None:
        out[0] = np.asarray(x0, dtype=float)
    for t in range(T - 1):
        x1, x2 = observer_step(lds, out[t, :n1], out[t, n1:], y_zscored[t], u_opto_zscored[t])
        out[t + 1] = np.concatenate([x1, x2])
    return out
