"""Attribution, routing, and counterfactual metrics for a control cell.

Everything here is offline: it reads a finished `CtrlData`, its NULL twin, and the fitted LDS,
and needs no controller change. That is deliberate — these are the measurements that decide
whether the P1 controller build is worth doing.

Three groups:

- **Model-internal** — `rho_*` on behavior and `pi_*` on the neural readout, read from the
  model's own block decomposition. Manipulation checks, not evidence: they say the model's
  bookkeeping is self-consistent.
- **Plant-anchored** — `pi_*_plant` from the measured twin difference against the NULL-trial
  geometry, plus the twin floor and shuffle null that give them a zero point. The only axis a
  unified-LDS model can also be scored on.
- **Naturalness** — `D_atypical` (plant) and `M_nat` (model). Reported at every cell because the
  routing shares say which coordinates the response occupies, not how far the system was pushed,
  and a controller can stay wholly inside the natural subspace at an amplitude the network never
  produces on its own.

The P2 counterfactual scores both model families' light-off free runs against the same NULL
twin. The free-run recursion comes from `src.ctrl._blocks`, so what is scored here is the
recursion the planner actually used.
"""
import numpy as np
from sklearn.metrics import r2_score

from src.ctrl._blocks import (
    closed_loop_blocks,
    is_subspace,
    lfp_readout_blocks,
    readout_blocks,
    task_input_block,
    unified_blocks,
)
from src.helpers import drop_constant_behavior
from usid.eval.attribution import (
    attribution_shares,
    build_null_geometry,
    free_run,
    free_run_natural,
    mahalanobis_deviation,
    plant_routing_shares,
    routing_shares,
    shuffle_null_shares,
)

LAMBDA_REF_FRACTIONS = (0.01, 0.1, 1.0)
"""Multipliers on the median NULL eigenvalue, spanning a decade either side of the middle value.
A `pi_2_plant` verdict that flips across these is not a verdict."""

EPSILON_FRAC = 1e-3
"""Ridge on `Σ_null`, as a fraction of its leading eigenvalue."""

N_SHUFFLE_DRAWS = 20
P2_OFFSET_FRACTIONS = (0.1, 0.3, 0.5, 0.7)
"""Freeze points within a trial, as fractions of trial length. The USID/BRAID counterfactual gap
is expected to widen with the offset — the longer stimulation has run before the freeze, the more
carried induced displacement a unified latent is wrong about."""

P2_HORIZON_FRAC = 0.25
"""Free-run length after the freeze, as a fraction of trial length."""


def compute_attribution_metrics(ctrl_data, null_ctrl_data, lds, eval_mask, rng):
    """All attribution/routing/naturalness metrics for one control cell.

    Args:
        ctrl_data: the stimulated `CtrlData`.
        null_ctrl_data: its NULL twin — same seed, same trial boundaries, no opto.
        lds: the fitted LDS the controller planned with.
        eval_mask: indices of in-trial samples.
        rng: `np.random.Generator` for the shuffle null.

    Returns:
        flat dict of metric name -> float, suitable for the summary row.
    """
    out = {}
    out.update(_plant_metrics(ctrl_data, null_ctrl_data, eval_mask, rng))
    if is_subspace(lds):
        out.update(_model_metrics(ctrl_data, null_ctrl_data, lds, eval_mask))
    out.update(_counterfactual_metrics(ctrl_data, null_ctrl_data, lds))
    return out


def _to_recorded_units(z_model, lds):
    """Map a model behavior readout into the units `ctrl_data.Z` is recorded in.

    `Xhat_lds` is the z-scored latent, so `Cz·x + d_z` is z-scored behavior, while `Z` and `Zref`
    were un-z-scored on the way into `CtrlData`. Applying the same `z_std`/`z_mean` here is what
    makes the two comparable.
    """
    return z_model * np.asarray(lds.z_std, dtype=float) + np.asarray(lds.z_mean, dtype=float)


def _plant_metrics(ctrl_data, null_ctrl_data, eval_mask, rng):
    """`pi_*_plant` against both `Σ_null` estimates and both nulls, plus `D_atypical`.

    Uses no model quantity, so it scores subspace and unified models on one axis.
    """
    X_ctrl = np.asarray(ctrl_data.X, dtype=float)[eval_mask]
    X_null = np.asarray(null_ctrl_data.X, dtype=float)[eval_mask]
    Z_null = drop_constant_behavior(np.asarray(null_ctrl_data.Z, dtype=float), "null Z")[eval_mask]
    delta_X = X_ctrl - X_null

    lam = np.clip(np.linalg.eigvalsh(np.cov(X_null, rowvar=False)), 0.0, None)
    lam_scale = float(np.median(lam))
    eps = EPSILON_FRAC * float(np.max(lam))

    out = {}
    for label, behavior in (("marginal", None), ("conditioned", Z_null)):
        for frac in LAMBDA_REF_FRACTIONS:
            geom = build_null_geometry(
                X_null, lambda_ref=frac * lam_scale, epsilon=eps, behavior_null=behavior
            )
            _, pi_2 = plant_routing_shares(delta_X, geom)
            out[f"pi_2_plant_{label}_lref{frac}"] = pi_2
            if frac == LAMBDA_REF_FRACTIONS[1]:
                out[f"pi_2_plant_{label}"] = pi_2
                out[f"null_gap_ratio_{label}"] = geom.spectral_gap[0]
                out[f"pi_2_plant_{label}_shuffle_null"] = float(np.mean(
                    shuffle_null_shares(delta_X, geom, N_SHUFFLE_DRAWS, rng)
                ))
                out[f"D_atypical_{label}"] = mahalanobis_deviation(
                    delta_X, _covariance_of(X_null, behavior), eps
                )
    out["delta_X_norm"] = float(np.linalg.norm(delta_X))
    out["light_energy"] = float(np.sum(np.asarray(ctrl_data.Uopto, dtype=float) ** 2))
    return out


def _model_metrics(ctrl_data, null_ctrl_data, lds, eval_mask):
    """`rho_*`, `pi_*`, `R2_attrib`, `E_nat`, and `M_nat` from the model's own blocks."""
    A_cl, _ = closed_loop_blocks(lds)
    Cz1, Cz2, d_z = readout_blocks(lds)
    Cy1, Cy2 = lfp_readout_blocks(lds)
    n1 = lds.A11.shape[0]

    x = np.asarray(ctrl_data.Xhat_lds, dtype=float)
    x_free = _per_trial_free_run(A_cl, x, ctrl_data.i_trial_boundaries)

    z_nat = _to_recorded_units(x[:, :n1] @ Cz1.T + d_z, lds)
    z_ind = x[:, n1:] @ Cz2.T * np.asarray(lds.z_std, dtype=float)   # a difference: no offset
    reference = drop_constant_behavior(np.asarray(ctrl_data.Zref, dtype=float), "ctrl Zref")

    dx1 = x[:, :n1] - x_free[:, :n1]
    dx2 = x[:, n1:] - x_free[:, n1:]

    rho_nat, rho_ind = attribution_shares(
        z_nat[eval_mask], z_ind[eval_mask], reference[eval_mask]
    )
    pi_1, pi_2 = routing_shares(
        (dx1 @ Cy1.T)[eval_mask], (dx2 @ Cy2.T)[eval_mask]
    )

    Z_ctrl = drop_constant_behavior(np.asarray(ctrl_data.Z, dtype=float), "ctrl Z")
    Z_null = drop_constant_behavior(np.asarray(null_ctrl_data.Z, dtype=float), "null Z")

    x1_null = np.asarray(null_ctrl_data.Xhat_lds, dtype=float)[:, :n1]
    P1 = _covariance_of(x1_null[eval_mask], None)
    ridge = EPSILON_FRAC * float(np.max(np.clip(np.linalg.eigvalsh(P1), 0.0, None)))

    z_nat_free = _to_recorded_units(x_free[:, :n1] @ Cz1.T + d_z, lds)
    return {
        "rho_nat": rho_nat,
        "rho_ind": rho_ind,
        "pi_1": pi_1,
        "pi_2": pi_2,
        # The model asserts how much it added to behavior. The twin reports how much actually
        # was added. Expected high at alpha=1 and low at alpha=0 — both low means attribution is
        # broken and the split is nominal.
        "R2_attrib": _safe_r2((Z_ctrl - Z_null)[eval_mask], z_ind[eval_mask]),
        # A penalized quantity: a check that the penalty bites, never evidence.
        "E_nat": float(
            np.sum((z_nat - z_nat_free)[eval_mask] ** 2)
            / max(np.sum(z_nat_free[eval_mask] ** 2), np.finfo(float).tiny)
        ),
        "M_nat": mahalanobis_deviation(dx1[eval_mask], P1, ridge),
    }


def _counterfactual_metrics(ctrl_data, null_ctrl_data, lds):
    """P2: each model's light-off free run scored against the NULL twin, by freeze offset.

    A subspace model gets two variants. `quarantine` discards `x̂2` at the freeze point ("what if
    the injection had never happened"), while `carry` keeps it and lets it decay into x1 through
    A12 ("what if I stop stimulating now"). A unified model has only `carry`.
    """
    Z_null = drop_constant_behavior(np.asarray(null_ctrl_data.Z, dtype=float), "null Z")
    x = np.asarray(ctrl_data.Xhat_lds, dtype=float)
    boundaries = np.asarray(ctrl_data.i_trial_boundaries)
    conditions = np.asarray(ctrl_data.conditions)
    trial_len = int(np.median(boundaries[:, 1] - boundaries[:, 0]))
    horizon = max(1, int(P2_HORIZON_FRAC * trial_len))
    B_task = task_input_block(lds)

    def drive(t0):
        """Modelled task drive over the counterfactual window, or None when the fit has no task path."""
        if B_task is None:
            return None
        # The recursion is x_{t+1} = A x_t + B u_t, so the step producing x_{t0+1} consumes u_{t0}.
        u = np.asarray(ctrl_data.Utask, dtype=float)[t0: t0 + horizon]
        return u @ B_task.T

    if is_subspace(lds):
        A_cl, _ = closed_loop_blocks(lds)
        Cz1, _, d_z = readout_blocks(lds)
        n1 = lds.A11.shape[0]
        variants = {
            "quarantine": lambda x0, t0: _to_recorded_units(
                free_run_natural(A_cl, x0, n1, horizon, True, drive(t0)) @ Cz1.T + d_z, lds),
            "carry": lambda x0, t0: _to_recorded_units(
                free_run_natural(A_cl, x0, n1, horizon, False, drive(t0)) @ Cz1.T + d_z, lds),
        }
    else:
        A, _ = unified_blocks(lds)
        Cz = np.asarray(lds.Cz, dtype=float)
        variants = {"carry": lambda x0, t0: _to_recorded_units(
            free_run(A, x0, horizon, drive(t0)) @ Cz.T, lds)}

    # Trivial baselines on the identical windows. A model free-run that loses to persistence has
    # a dynamics problem; one that loses to the trial mean is not predicting at all, and neither
    # verdict is visible from R2_cf alone.
    baselines = {
        "persistence": lambda x0, t0: np.repeat(Z_null[t0][None, :], horizon, axis=0),
        "trialmean": lambda x0, t0: np.repeat(
            Z_null[max(t0 - horizon, 0): t0 + 1].mean(axis=0)[None, :], horizon, axis=0
        ),
    }

    out = {}
    for name, predict in {**variants, **baselines}.items():
        for frac in P2_OFFSET_FRACTIONS:
            predicted, observed = [], []
            for start, end in boundaries:
                t0 = int(start + frac * (end - start))
                if t0 + 1 + horizon > end:
                    continue
                predicted.append(predict(x[t0], t0))
                observed.append(Z_null[t0 + 1: t0 + 1 + horizon])
            if not predicted:
                out[f"R2_cf_{name}_tau{frac}"] = np.nan
                continue
            out[f"R2_cf_{name}_tau{frac}"] = _safe_r2(
                np.concatenate(observed), np.concatenate(predicted)
            )
    out.update(_ceiling_scores(Z_null, boundaries, conditions, horizon))
    for name in list(variants) + list(baselines):
        for frac in P2_OFFSET_FRACTIONS:
            ceiling = out.get(f"R2_cf_ceiling_tau{frac}", np.nan)
            got = out.get(f"R2_cf_{name}_tau{frac}", np.nan)
            # Fraction of the achievable predictability captured. 1.0 matches the ceiling, 0 matches
            # the window mean, negative is worse than the mean. A raw R2 cannot distinguish a bad
            # model from an unpredictable window; this can.
            out[f"R2_cf_{name}_vs_ceiling_tau{frac}"] = (
                got / ceiling if np.isfinite(ceiling) and ceiling > 0 else np.nan
            )
    out["R2_cf_offset_slope"] = _offset_slope(out, list(variants)[0])
    return out


def _ceiling_scores(Z_null, boundaries, conditions, horizon):
    """Leave-one-out condition-average predictor: the achievable R² knowing only the task.

    Averages the same within-trial window over the other trials of the same reach, so it carries no
    information about neural state or stimulation. What it scores is how much of the behavior in
    this window is task-stereotyped, which is the ceiling any single-trial predictor is working
    against.
    """
    out = {}
    for frac in P2_OFFSET_FRACTIONS:
        predicted, observed = [], []
        starts = {i: int(s + frac * (e - s)) for i, (s, e) in enumerate(boundaries)}
        for i, (s, e) in enumerate(boundaries):
            t0 = starts[i]
            if t0 + 1 + horizon > e:
                continue
            peers = [
                Z_null[starts[j] + 1: starts[j] + 1 + horizon]
                for j, (s2, e2) in enumerate(boundaries)
                if j != i and conditions[j] == conditions[i]
                and starts[j] + 1 + horizon <= e2
            ]
            peers = [p for p in peers if p.shape == (horizon, Z_null.shape[1])]
            if not peers:
                continue
            predicted.append(np.mean(peers, axis=0))
            observed.append(Z_null[t0 + 1: t0 + 1 + horizon])
        out[f"R2_cf_ceiling_tau{frac}"] = (
            _safe_r2(np.concatenate(observed), np.concatenate(predicted))
            if predicted else np.nan
        )
    return out


def _per_trial_free_run(A_cl, x, boundaries):
    """Light-off free run from each trial's first sample, filling that trial's rows.

    The deviation `Δx` the routing shares are built on is measured against what the model would
    have done from the start of the trial with the light off, so the reference restarts at every
    trial boundary rather than running once across the session.
    """
    out = np.zeros_like(x)
    for start, end in boundaries:
        steps = int(end - start)
        out[start] = x[start]
        if steps > 1:
            out[start + 1: end] = free_run(A_cl, x[start], steps - 1)
    return out


def _covariance_of(data, behavior):
    """Covariance of `data`, optionally after regressing out `behavior`."""
    centered = data - data.mean(axis=0, keepdims=True)
    if behavior is not None:
        design = np.column_stack([behavior, np.ones(behavior.shape[0])])
        coef, *_ = np.linalg.lstsq(design, centered, rcond=None)
        centered = centered - design @ coef
    return np.atleast_2d(np.cov(centered, rowvar=False))


def _offset_slope(scores, variant):
    """Least-squares slope of `R2_cf` against freeze offset, for one variant.

    The level of a single `R2_cf` is confounded by how well either model tracks. The trend across
    offsets is the discriminating signature, so it is reported as its own column.
    """
    xs, ys = [], []
    for frac in P2_OFFSET_FRACTIONS:
        value = scores.get(f"R2_cf_{variant}_tau{frac}", np.nan)
        if np.isfinite(value):
            xs.append(frac)
            ys.append(value)
    if len(xs) < 2:
        return np.nan
    return float(np.polyfit(xs, ys, 1)[0])


def _safe_r2(y_true, y_pred):
    """R² averaged over the output dimensions with finite scores.

    Mirrors `analyze_ctrl_data.r2_score_masked`: `force_finite=False` lets a constant true series
    come back as -inf rather than silently collapsing to 0 or 1 and shifting the average.
    """
    if np.size(y_true) == 0 or np.size(y_pred) == 0:
        return np.nan
    try:
        per_dim = r2_score(
            y_true, y_pred, multioutput="raw_values", force_finite=False
        )
    except ValueError:
        return np.nan
    finite = np.isfinite(per_dim)
    if not finite.any():
        return np.nan
    return float(np.mean(per_dim, where=finite))
