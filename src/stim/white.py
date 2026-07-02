"""Legacy white-lognormal sys-ID stim (restored as a selectable method).

Per-channel, per-step i.i.d. lognormal amplitude (median = typical_Irr0, logstd =
log(u_max)/3, clipped to u_max) gated by an independent Bernoulli. Temporally WHITE
(flat spectrum) -- this is the structure the OED replaced. The Bernoulli on-probability
(duty) is set so the expected mean light equals the budget, so it sits at the SAME
operating point as the OED design: a clean white-vs-OED ablation at matched budget,
isolating the operating point from the OED's pulsed/graded structure.

(The exact pre-OED default was cap=30, active_frac=0.5; recover it by setting
stim_u_max_mWmm2=30 and the budget to that corner's mean.)
"""
import numpy as np

# Fixed RNG for the one-time duty calibration so it never perturbs the stim stream.
_DUTY_EST_SEED = 12345
_DUTY_EST_N = 200_000


class WhiteLognormalSampler:
    def __init__(self, n_channels, u_max, budget_mean, typical_Irr0, command_dt_ms, rng):
        self.k = int(n_channels)
        self.u_max = float(u_max)
        self.rng = rng
        self.logmean = np.log(float(typical_Irr0))
        self.logstd = np.log(float(u_max)) / 3.0
        # Duty so E[active light] == budget: mean = active_frac * E[clipped lognormal].
        est = np.random.default_rng(_DUTY_EST_SEED).lognormal(
            self.logmean, self.logstd, _DUTY_EST_N).clip(0.0, self.u_max)
        self.active_frac = float(np.clip(budget_mean / est.mean(), 0.0, 1.0))

    def step(self):
        amp = self.rng.lognormal(self.logmean, self.logstd, self.k).clip(0.0, self.u_max)
        return amp * (self.rng.random(self.k) < self.active_frac)
