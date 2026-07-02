"""Frequency-swept (chirp) sys-ID stim at a fixed operating point (a selectable method).

Each channel is an independent log-frequency sweep (F0..F1 Hz over T_SWEEP, repeating) on
a positive offset so the mean light ~ budget:
    light(t) = clip(offset + amp*sin(phase), 0, u_max),  offset = amp = budget.
This is the classic broadband linear-ID input -- it covers the identifiable band
(~0.05-11 Hz) via the sweep. But because light is non-negative it must ride on a positive
offset, so at a low budget its amplitude is bounded (~2*budget) and it CANNOT probe the
opsin's high-amplitude/saturating regime (unlike the OED's sparse high-amplitude bursts).
Included as an ablation reference (the "just sweep the frequencies" intuition); the offline
Fisher analysis shows it ties white and trails OED at a matched budget.
"""
import numpy as np

F0_HZ, F1_HZ, T_SWEEP_MS = 0.1, 10.0, 5000.0


class ChirpSampler:
    def __init__(self, n_channels, u_max, budget_mean, command_dt_ms, rng):
        self.k = int(n_channels)
        self.u_max = float(u_max)
        self.dt_s = float(command_dt_ms) / 1000.0
        self.offset = float(budget_mean)
        self.amp = float(budget_mean)
        self._t_sweep = T_SWEEP_MS / 1000.0
        # Decorrelate channels: random phase + staggered sweep position.
        self._phase = rng.uniform(0.0, 2.0 * np.pi, self.k)
        self._t = rng.uniform(0.0, self._t_sweep, self.k)

    def step(self):
        frac = (self._t % self._t_sweep) / self._t_sweep
        f = F0_HZ * (F1_HZ / F0_HZ) ** frac          # log sweep, per channel (Hz)
        self._phase = self._phase + 2.0 * np.pi * f * self.dt_s
        self._t = self._t + self.dt_s
        light = self.offset + self.amp * np.sin(self._phase)
        return np.clip(light, 0.0, self.u_max)
