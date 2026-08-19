"""Opsin-NN training light distributions, selected by the ``opsin_stim_method`` wildcard.

Each method builds a single ``(time_steps,)`` commanded-light trace (mW/mm^2, clipped to
``[0, u_max]``) that the ``opsin_train_cold`` rule feeds through the single-neuron Brian2
ChR2 sim to produce the paired photocurrent target. Everything downstream of the trace (the
Brian2 sim, the NN fit) is distribution-agnostic, so a new training set = a new generator
here. All methods share the actuator cap ``u_max`` so the NN covers the same input range.

Two options (a sweep over `opsin_stim_method` found the opsin training distribution does NOT
affect closed-loop control Q — see cold_opsinstim_sweep_results_20260707; both give
near-identical opsin fidelity and Q, so the choice is a parsimony/coverage preference):

  current_mix : 60% sustained on/off blocks amp~U(0,u_max) + 40% white lognormal. The
                validated default (the +0.305 headline opsin); explicit sustained-high blocks
                give the best deep-saturation coverage, sustained-zero blocks pin Iopto(0)=0.
  ou          : single clipped Ornstein-Uhlenbeck colored-noise process (tau ~ Gd2). One
                continuous distribution (3 knobs vs current_mix's ~8); ramps smoothly through
                all amplitudes (best low/operating-band fidelity), but under-covers the
                sustained extremes vs current_mix's blocks.
  logwalk     : piecewise-constant light with log-uniform dwell and a log-amplitude random
                walk. Collapses current_mix's block/white pair into one leaf: dwell 1 is the
                i.i.d. white limit, dwell ~dwell_max the sustained-block limit, and every
                dwell in between is reachable. Correlated successive amplitudes additionally
                produce small steps away from a settled baseline, which neither leaf of
                current_mix generates (its median ON-block transition is tens of mW/mm^2).
"""
import numpy as np

OPSIN_STIM_METHODS = ("current_mix", "ou", "logwalk")


def _block_trace(rng, time_steps, u_max, amp_max, sustain_min, sustain_max,
                 off_min, off_max, on_prob):
    """Piecewise-constant on/off block light. ON blocks hold a constant amplitude
    ~U(0, amp_max) for sustain_min..sustain_max steps (sustained drive spanning up to
    amp_max — reaches and holds the saturating regime the pure-white trace never sustains,
    with uniform amplitude for even steady-state-curve coverage). OFF blocks hold zero for
    off_min..off_max steps — sustained zero, pinning Iopto(0)=0 and exercising the recovery
    kinetics in the gaps."""
    trace = np.zeros(time_steps, dtype=np.float64)
    pos = 0
    while pos < time_steps:
        if rng.random() < on_prob:
            L = int(rng.integers(sustain_min, sustain_max + 1))
            amp = float(min(rng.uniform(0.0, amp_max), u_max))
            trace[pos:pos + L] = amp
        else:
            L = int(rng.integers(off_min, off_max + 1))
        pos += L
    return trace[:time_steps]


def _white_trace(rng, time_steps, u_max, typical_Irr0):
    """White lognormal light, i.i.d. per step (the 40% low-range component of current_mix).
    Densely samples the low/controller range but rarely sustains high irradiance, so it
    under-covers deep saturation on its own — hence the block component."""
    mu = float(np.log(typical_Irr0))
    sigma = float(np.log(u_max) / 3.0)
    return rng.lognormal(mu, sigma, size=time_steps).clip(0.0, u_max)


def _ou_trace(rng, time_steps, u_max, dt_ms, mu, tau_ms, std):
    """Clipped Ornstein-Uhlenbeck colored-noise light: dX = -(X-mu)/tau dt + sigma dW,
    clipped to [0, u_max]. Continuous amplitude ramps smoothly through 0..u_max (best
    Hill-curve + slope coverage); tau ~ Gd2 (~100 ms) so levels linger long enough to engage
    desensitization. sigma is set from the target stationary std (Var = sigma^2 * tau / 2)."""
    x = np.empty(time_steps, dtype=np.float64)
    x[0] = float(mu)
    sigma = float(std) * np.sqrt(2.0 / float(tau_ms))
    dW = rng.normal(0.0, np.sqrt(float(dt_ms)), size=time_steps)
    for t in range(1, time_steps):
        x[t] = x[t - 1] - (x[t - 1] - mu) / tau_ms * dt_ms + sigma * dW[t]
    return np.clip(x, 0.0, u_max)


def _logwalk_trace(rng, time_steps, u_max, amp_min, amp_max, dwell_max,
                   reset_prob, step_sigma, off_prob):
    """Piecewise-constant light whose segment dwell is log-uniform on [1, dwell_max] and
    whose amplitude follows a random walk in log space. Each segment is either exact zero
    (prob off_prob, pinning Iopto(0)=0 and exercising recovery kinetics), a fresh
    log-uniform draw over [amp_min, amp_max] (prob reset_prob, covering the full actuator
    range), or the previous amplitude scaled by exp(N(0, step_sigma)).

    The scaled branch is the one current_mix cannot produce: successive levels are
    correlated, so some transitions are small *relative* steps, which at a low baseline are
    small *absolute* steps. Those identify the incremental gain d(Iopto)/d(light) near zero
    light, which needs a settled baseline plus a small perturbation. Sampling low amplitudes
    is not sufficient on its own -- current_mix's white leaf already puts ~32% of its samples
    below 2.63 mW/mm^2, i.i.d., and identifies nothing there because it never settles."""
    log_lo, log_hi = float(np.log(amp_min)), float(np.log(amp_max))
    log_dwell = float(np.log(dwell_max))
    trace = np.zeros(time_steps, dtype=np.float64)
    pos = 0
    log_a = rng.uniform(log_lo, log_hi)
    while pos < time_steps:
        L = max(1, int(round(float(np.exp(rng.uniform(0.0, log_dwell))))))
        if rng.random() < off_prob:
            amp = 0.0
        else:
            if rng.random() < reset_prob:
                log_a = rng.uniform(log_lo, log_hi)
            else:
                log_a = float(np.clip(log_a + rng.normal(0.0, step_sigma), log_lo, log_hi))
            amp = float(np.exp(log_a))
        trace[pos:pos + L] = min(amp, u_max)
        pos += L
    return trace[:time_steps]


def _resolve_leaf(method, seq_idx, n_sequences, params):
    """Map a top-level method to the leaf generator for sequence ``seq_idx``. ``current_mix``
    is a per-sequence block/white mixture (by the const_block_fraction partition, reproducing
    the validated default exactly); ``ou`` is a single distribution applied to all sequences."""
    if method == "current_mix":
        n_const = int(round(params["const_block_fraction"] * n_sequences))
        return "block" if seq_idx < n_const else "white"
    if method == "ou":
        return "ou"
    if method == "logwalk":
        return "logwalk"
    raise ValueError(
        f"unknown opsin_stim_method {method!r}; valid: {OPSIN_STIM_METHODS}")


def build_opsin_light(method, rng, time_steps, u_max, params, seq_idx=0, n_sequences=1):
    """Return one ``(time_steps,)`` commanded-light trace in [0, u_max] mW/mm^2 for
    opsin-NN training. ``params`` carries the per-method knobs (see the config
    ``opsin_train_cold`` block). ``seq_idx``/``n_sequences`` are only used by
    ``current_mix`` to place its block-vs-white slot."""
    leaf = _resolve_leaf(method, seq_idx, n_sequences, params)
    if leaf == "block":
        trace = _block_trace(
            rng, time_steps, u_max, params["sustain_amp_max"],
            params["sustain_steps_min"], params["sustain_steps_max"],
            params["off_steps_min"], params["off_steps_max"],
            params["block_on_probability"])
    elif leaf == "white":
        trace = _white_trace(rng, time_steps, u_max, params["typical_Irr0_mWmm2"])
    elif leaf == "logwalk":
        trace = _logwalk_trace(
            rng, time_steps, u_max, params["logwalk_amp_min_mWmm2"],
            params["logwalk_amp_max_mWmm2"], params["logwalk_dwell_max_steps"],
            params["logwalk_reset_probability"], params["logwalk_step_sigma"],
            params["logwalk_off_probability"])
    elif leaf == "ou":
        trace = _ou_trace(
            rng, time_steps, u_max, params["command_dt_ms"], params["ou_mu_mWmm2"],
            params["ou_tau_ms"], params["ou_std_mWmm2"])
    else:  # unreachable — _resolve_leaf validates
        raise ValueError(f"unknown opsin light leaf {leaf!r}")
    return np.clip(trace, 0.0, u_max).astype(np.float64)
