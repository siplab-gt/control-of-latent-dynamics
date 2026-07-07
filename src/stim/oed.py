"""Optimal-experiment-design (OED) of the sys-ID opto stimulation.

Derives the light stimulation that identifies the opsin best, from first principles
rather than by imitating a controller: choose the stim that maximizes the Fisher
information about the opsin's light->photocurrent map, subject to the actuator
constraints (light in [0, u_max]) and a mean-light budget that fixes the operating
point. The budget is the lever that encodes the de-saturation finding -- without it,
maximizing information just blasts maximum light (more signal -> more info) and parks
the opsin in chronic desensitization, exactly the failure sparse stim avoided. At a
fixed (low, deployment-matched) mean budget, the information-optimal design is forced
to spend its light wisely, and it comes out pulsed (recovery gaps so the desensitized
pools relax) and -- in the low, deployment-range budget regime -- graded in amplitude
(to map the Hill curve); at high budgets the optimum collapses toward constant
near-u_max light. The structure is derived, not imposed. NOTE: this maximizes the
information about the opsin observed in isolation (photocurrent directly measured); in
the full pipeline photocurrent is latent behind B' and the LFP, so D-optimality here
is a principled proxy for de-saturating, range-spanning excitation -- the downstream
control metrics, not the D-optimal label, are the judge.

Cramer-Rao: the Fisher information matrix M is the inverse of the best achievable
parameter-estimate covariance, so maximizing a scalar of M (here log det M, the
D-optimal criterion) directly minimizes the unavoidable identification uncertainty
for a fixed experimental cost. Sensitivities dIopto/dtheta are taken by central
finite differences through the validated ``OpsinSurrogate`` (one source of truth for
the opsin; no hand-derived variational ODEs).

The opsin is per-channel and channels do not overlap, so the per-channel Fisher
information is additive across channels and the design problem is single-channel: we
optimize one channel's renewal-process law and apply it independently per channel.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution

from src.stim.opsin_surrogate import OpsinSurrogate, load_calibrated_surrogate

# Control-relevant parameter subset to identify: conductance scale, Hill
# half-saturation (where authority bends over), and the medium desensitization rate
# Gd2 (tau ~ 100 ms) that sets the gain droop within a control window. Gr0
# (tau ~ 3 s) is intentionally excluded: the controller's ~820 ms windows only barely
# engage that slow pool (~24% turnover), Gr0 has by far the lowest Fisher sensitivity
# of the candidates, and identifying it would demand budget-blowing seconds-long gaps.
DEFAULT_ID_PARAMS = ("g0", "phim_irr", "Gd2")

# Surrogate integrates at this fine step; commands are held piecewise-constant over
# the (coarser) control/command period, matching how cleo applies the light array.
FINE_DT_MS = 1.0

# Amplitude levels in the deterministic representative probe used by the optimizer
# objective (see ``representative_waveform``). More levels = finer curve coverage.
_PROBE_N_LEVELS = 5
# Cap a single recovery gap in the probe. 400 ms is several Gd2 time constants
# (tau ~ 100 ms), enough to score the medium-desensitization recovery the controller
# experiences, and keeps the objective integration tractable.
_PROBE_MAX_GAP_MS = 400.0
_FIM_SEED = 20260627


def _command_to_fine(command, command_dt_ms, fine_dt_ms=FINE_DT_MS):
    """Upsample a piecewise-constant command sequence to the fine integration grid."""
    reps = int(round(command_dt_ms / fine_dt_ms))
    return np.repeat(np.asarray(command, dtype=np.float64), reps)


_FIM_SUBSTEPS = 1  # forward-Euler sub-step for the FD sensitivity integration
# (1 ms steps are stable for these rates, max ~0.4/ms; accuracy is ample for FD)


def _simulate_batch(light_fine, g0, phim_irr, k1, k2, Gf0, kf, Gb0, kb,
                    Gd1, Gd2, Gr0, gamma, E, v, dt=FINE_DT_MS, substeps=_FIM_SUBSTEPS):
    """Integrate B four-state opsins (each a (B,) vector of params) under the same
    light_fine, in one vectorized time loop. Returns Iopto of shape (T, B). Same ODEs
    as OpsinSurrogate.simulate (verified equal at B=1 in tests); batching the FD
    parameter perturbations collapses 2*len(id_params) sims into one loop."""
    B = np.broadcast(g0, phim_irr, k1, Gd2, Gr0).shape[0]
    g0, phim_irr = np.broadcast_to(g0, (B,)), np.broadcast_to(phim_irr, (B,))
    k1, k2 = np.broadcast_to(k1, (B,)), np.broadcast_to(k2, (B,))
    Gf0, kf = np.broadcast_to(Gf0, (B,)), np.broadcast_to(kf, (B,))
    Gb0, kb = np.broadcast_to(Gb0, (B,)), np.broadcast_to(kb, (B,))
    Gd1, Gd2 = np.broadcast_to(Gd1, (B,)), np.broadcast_to(Gd2, (B,))
    Gr0, gamma = np.broadcast_to(Gr0, (B,)), np.broadcast_to(gamma, (B,))
    scale = -g0 * (v - E)
    h = dt / substeps
    C1 = np.ones(B); O1 = np.zeros(B); O2 = np.zeros(B)
    out = np.empty((light_fine.shape[0], B), dtype=np.float64)
    for s in range(light_fine.shape[0]):
        L = light_fine[s]
        H = L / (L + phim_irr) if L > 0 else np.zeros(B)
        a1, a2 = k1 * H, k2 * H
        gf, gb = kf * H + Gf0, kb * H + Gb0
        for _ in range(substeps):
            C2 = 1.0 - C1 - O1 - O2
            dC1 = Gd1 * O1 + Gr0 * C2 - a1 * C1      # simultaneous (Jacobi) update,
            dO1 = a1 * C1 + gb * O2 - (Gd1 + gf) * O1  # all deltas from the old state,
            dO2 = a2 * C2 + gf * O1 - (Gd2 + gb) * O2  # to match OpsinSurrogate.simulate
            C1 = C1 + h * dC1
            O1 = O1 + h * dO1
            O2 = O2 + h * dO2
        out[s] = scale * (O1 + gamma * O2)
    return out


def fisher_logdet(command, surrogate, command_dt_ms,
                  id_params=DEFAULT_ID_PARAMS, noise_std=0.02, rel_eps=1e-3):
    """log det of the Fisher information matrix for identifying ``id_params`` from the
    opsin response to ``command`` (commanded light per command step, mW/mm^2).

    Sensitivities are central finite differences in each parameter; M = S^T S /
    noise_std^2 with S[t, j] = dIopto(t)/dtheta_j. A small ridge keeps slogdet finite
    when a parameter is locally unidentifiable."""
    light_fine = _command_to_fine(command, command_dt_ms)
    base = {n: float(getattr(surrogate, n)) for n in OpsinSurrogate.PARAM_NAMES}
    P = len(id_params)
    arrs = {n: np.full(2 * P, base[n], dtype=np.float64)
            for n in OpsinSurrogate.PARAM_NAMES}
    deltas = np.empty(P, dtype=np.float64)
    for j, name in enumerate(id_params):
        theta = base[name]
        d = rel_eps * abs(theta) if theta != 0.0 else rel_eps
        deltas[j] = d
        arrs[name][2 * j] = theta + d
        arrs[name][2 * j + 1] = theta - d
    iopto = _simulate_batch(light_fine, E=surrogate.E, v=surrogate.v, **arrs)
    S = np.empty((light_fine.shape[0], P), dtype=np.float64)
    for j in range(P):
        S[:, j] = (iopto[:, 2 * j] - iopto[:, 2 * j + 1]) / (2.0 * deltas[j])
    M = S.T @ S / (noise_std ** 2)
    sign, logdet = np.linalg.slogdet(M + 1e-9 * np.eye(P))
    return float(logdet) if sign > 0 else -np.inf


@dataclass(frozen=True)
class StimDesign:
    """OED-derived per-channel renewal law for the sys-ID opto stim.

    A channel alternates ON bursts (held amplitude drawn uniformly in
    [amp_lo, amp_hi]) of mean duration ``mean_on_ms`` with OFF gaps of mean duration
    ``mean_off_ms``; both run-lengths are exponential (memoryless renewal -> broadband
    via the on/off edges). All emitted light is in [0, u_max] mW/mm^2 and the
    expected mean light equals ``expected_mean_light()``.
    """
    mean_on_ms: float
    mean_off_ms: float
    amp_lo: float
    amp_hi: float
    u_max: float
    command_dt_ms: float

    def __post_init__(self):
        # Enforce amp_lo <= amp_hi so a directly-constructed design never passes
        # low > high to rng.uniform in the sampler (frozen dataclass -> setattr).
        if self.amp_hi < self.amp_lo:
            lo, hi = self.amp_hi, self.amp_lo
            object.__setattr__(self, "amp_lo", lo)
            object.__setattr__(self, "amp_hi", hi)

    @property
    def duty(self):
        return self.mean_on_ms / (self.mean_on_ms + self.mean_off_ms)

    def expected_mean_light(self):
        return self.duty * 0.5 * (self.amp_lo + self.amp_hi)

    def make_sampler(self, n_channels, rng):
        """Stateful per-channel renewal sampler. Call ``.step()`` once per command
        step to get an (n_channels,) light array; independent across channels."""
        return _RenewalSampler(self, n_channels, rng)

    def representative_waveform(self, n_levels=_PROBE_N_LEVELS):
        """Deterministic probe capturing the design's information content without
        renewal sampling noise: ``n_levels`` held ON bursts whose amplitudes tile
        [amp_lo, amp_hi] (curve coverage), each of duration ``mean_on_ms`` followed by
        an OFF gap of ``min(mean_off_ms, _PROBE_MAX_GAP_MS)`` (kinetics + recovery).
        Returns the per-command-step light sequence."""
        on_steps = max(1, int(round(self.mean_on_ms / self.command_dt_ms)))
        gap_steps = max(1, int(round(min(self.mean_off_ms, _PROBE_MAX_GAP_MS)
                                     / self.command_dt_ms)))
        amps = (np.full(1, self.amp_lo) if n_levels == 1
                else np.linspace(self.amp_lo, self.amp_hi, n_levels))
        cmd = []
        for a in amps:
            cmd.extend([min(a, self.u_max)] * on_steps)
            cmd.extend([0.0] * gap_steps)
        return np.array(cmd, dtype=np.float64)


class _RenewalSampler:
    """Per-channel renewal state advanced one command step at a time."""

    def __init__(self, design, n_channels, rng):
        self.d = design
        self.k = int(n_channels)
        self.rng = rng
        self.dt = design.command_dt_ms
        self._on = rng.random(self.k) < design.duty
        self._amp = np.where(self._on,
                             rng.uniform(design.amp_lo, design.amp_hi, self.k), 0.0)
        self._rem = self._draw_runs(self._on)

    def _draw_runs(self, on_mask):
        # Round (not ceil+1) so the realized run lengths are unbiased: the deployed
        # mean light then matches StimDesign.expected_mean_light() (the optimizer's
        # budget), instead of overshooting it ~12%. Floor at one command step.
        means = np.where(on_mask, self.d.mean_on_ms, self.d.mean_off_ms)
        return np.maximum(1, np.round(self.rng.exponential(means) / self.dt)).astype(int)

    def step(self):
        expired = self._rem <= 0
        if np.any(expired):
            self._on[expired] = ~self._on[expired]
            new_amp = np.where(self._on[expired],
                               self.rng.uniform(self.d.amp_lo, self.d.amp_hi,
                                                int(expired.sum())), 0.0)
            self._amp[expired] = new_amp
            self._rem[expired] = self._draw_runs(self._on[expired])
        out = np.clip(self._amp.copy(), 0.0, self.d.u_max)
        self._rem -= 1
        return out


def optimize_design(surrogate, u_max, budget_mean, command_dt_ms,
                    id_params=DEFAULT_ID_PARAMS, noise_std=0.02,
                    seed=_FIM_SEED, maxiter=30):
    """Optimize the renewal law subject to expected mean light <= budget_mean and
    amplitudes in [0, u_max], maximizing the opsin Fisher log-det (D-optimal, parameter
    precision). Corner-seeking -> high amplitude at the budget.

    differential_evolution over the 4 design params. The objective scores each candidate on
    its deterministic representative probe (``StimDesign.representative_waveform``) so it is
    noise-free and reproducible. Returns a ``StimDesign``; deterministic given (surrogate,
    constraints, seed)."""
    bounds = [
        (command_dt_ms, 400.0),          # mean_on_ms: >= one command step
        (command_dt_ms, 800.0),          # mean_off_ms (few x Gd2 recovery)
        (0.0, u_max),                    # amp_lo
        (0.0, u_max),                    # amp_hi
    ]

    def objective(x):
        mean_on, mean_off, amp_lo, amp_hi = x
        if amp_hi < amp_lo:
            amp_lo, amp_hi = amp_hi, amp_lo
        design = StimDesign(mean_on, mean_off, amp_lo, amp_hi, u_max, command_dt_ms)
        over = design.expected_mean_light() - budget_mean
        if over > 0:                     # budget violation: smooth penalty
            return 1e3 + 1e3 * over
        cmd = design.representative_waveform()
        if cmd.max() <= 0:               # all-dark design is uninformative
            return 1e3
        return -fisher_logdet(cmd, surrogate, command_dt_ms,
                              id_params=id_params, noise_std=noise_std)

    result = differential_evolution(
        objective, bounds, seed=seed, maxiter=maxiter, tol=1e-3,
        popsize=8, polish=True, init="sobol",
    )
    mean_on, mean_off, amp_lo, amp_hi = result.x
    if amp_hi < amp_lo:
        amp_lo, amp_hi = amp_hi, amp_lo
    return StimDesign(float(mean_on), float(mean_off), float(amp_lo), float(amp_hi),
                      float(u_max), float(command_dt_ms))


_DESIGN_CACHE = {}


def derive_optimal_design(u_max, budget_mean, command_dt_ms,
                          calibration_json=None, **kwargs):
    """Cached OED design for the calibrated plant. Pure function of the calibration +
    constraints, so it is computed once and reused across all sys-ID cells."""
    surrogate = (load_calibrated_surrogate(calibration_json) if calibration_json
                 else load_calibrated_surrogate())
    key = (round(u_max, 6), round(budget_mean, 6), round(command_dt_ms, 6),
           surrogate.phim_irr, surrogate.g0, tuple(sorted(kwargs.items())))
    if key not in _DESIGN_CACHE:
        _DESIGN_CACHE[key] = optimize_design(
            surrogate, u_max, budget_mean, command_dt_ms, **kwargs)
    return _DESIGN_CACHE[key]
