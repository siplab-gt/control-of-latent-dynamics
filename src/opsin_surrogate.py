"""Fast deterministic numpy surrogate of the calibrated UnitlessChr2FourState opsin.

The fSNN opsin (``src.opsins.UnitlessChr2FourState``, a Bansal-2020 four-state ChR2)
is the only nonlinearity on the light->photocurrent path and the thing whose
mis-identification drives the controller into over-saturation. Optimal-experiment
design for the sys-ID stim (``src.opsin_oed``) needs to evaluate that opsin's
response to thousands of candidate light waveforms, which is far too expensive
through Brian2. This module integrates the *same* four-state ODEs in vectorized
numpy, deterministically, in milliseconds.

One source of truth: the ODE structure and kinetic constants are copied verbatim
from ``src.opsins.UnitlessChr2FourState.model`` and cleo's ``BansalFourStateOpsin``
defaults. The single fitted quantity, ``phim_irr`` (the Hill half-saturation
expressed in *commanded* mW/mm^2 rather than cleo's internal photon flux), is
derived from the committed calibration curve so the surrogate tracks the plant:
``derive_phim_irr`` fits it to ``chr2_4s_fsnn_saturation_curve.json`` and the
surrogate then reproduces all 15 points of that curve to <0.05% (validated in the USID
repo, ``tests/test_opsin_oed.py::TestOpsinSurrogate``).

Like the rest of the calibration, this is the *representative* opsin: ``g0`` and the
fitted ``phim_irr`` correspond to the 500 um calibration reference distance, not the
per-neuron spread of effective saturation across the fSNN population.

Why commanded units: cleo's ``phim`` is a photon flux, and the opsin sees
``transmittance * commanded_light`` at the calibration reference distance. Both
the light->photon-flux conversion and the transmittance are linear, so in
commanded-light units the Hill term stays ``light/(light+phim_irr)`` with a single
effective ``phim_irr`` that absorbs both constants. The sys-ID stim is commanded
in mW/mm^2, so this is exactly the space the design lives in.

The calibration bench reads "steady state" as the mean of the last 10 samples of a
500 ms pulse (``calibrate_opsin_plant.DEFAULT_PULSE_DURATION_MS``). At 500 ms the
slow desensitized pool (Gr0, tau ~ 3 s) is only ~15% equilibrated, so that curve is
*not* the true asymptote; ``steady_state_curve`` reproduces the bench protocol for
validation, while ``simulate`` integrates the genuine transient the OED cares about.
"""
import functools
import json
import os

import numpy as np
from scipy.optimize import minimize_scalar

# Kinetic constants (1/ms) copied from cleo BansalFourStateOpsin defaults, which
# UnitlessChr2FourState inherits unchanged. p == q == 1 for Bansal, so the two
# Hill terms (Hp driving Ga1/Ga2, Hq driving Gf/Gb) collapse to one: H = phi/(phi+phim).
_BANSAL_KINETICS = dict(
    k1=0.4, k2=0.12, Gf0=0.018, kf=0.01, Gb0=0.008, kb=0.008,
    Gd1=0.066, Gd2=0.01, Gr0=3.33e-4, gamma=0.05,
)

# Bench protocol (must match calibrate_opsin_plant): steady state = mean of the
# last _BENCH_TAIL_MS of a _BENCH_PULSE_MS constant pulse.
_BENCH_PULSE_MS = 500.0
_BENCH_TAIL_MS = 10.0

# realpath (not abspath): Snakemake runs rules with `workflow/src` symlinked to
# ../src, so abspath(__file__) keeps the symlinked workflow/src path and resolves
# the repo root to workflow/. realpath follows the symlink to the real COLD root.
_DEFAULT_CALIBRATION_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "results", "opsin_calibration", "chr2_4s_fsnn_saturation_curve.json",
)


class OpsinSurrogate:
    """Deterministic four-state ChR2 surrogate in commanded-light (mW/mm^2) units.

    Parameters mirror the calibrated plant: ``g0`` (conductance scale), ``E`` and
    ``v`` (driving force, Iopto = -g0*fphi*(v-E)), the Bansal kinetic rates, and the
    fitted ``phim_irr``. ``simulate`` integrates the transient under an arbitrary
    light(t); ``steady_state_curve`` reproduces the calibration bench for validation.
    """

    # Parameters exposed to the OED sensitivity computation. Names match attributes.
    PARAM_NAMES = ("g0", "phim_irr", "k1", "k2", "Gf0", "kf", "Gb0", "kb",
                   "Gd1", "Gd2", "Gr0", "gamma")

    def __init__(self, g0, E, v, phim_irr, **kinetics):
        self.g0 = float(g0)
        self.E = float(E)
        self.v = float(v)
        self.phim_irr = float(phim_irr)
        for name, val in _BANSAL_KINETICS.items():
            setattr(self, name, float(kinetics.get(name, val)))

    @property
    def v_minus_E(self):
        return self.v - self.E

    def _rates(self, light_t):
        """Light-dependent transition rates (1/ms) for each step of light_t."""
        H = np.where(light_t > 0.0, light_t / (light_t + self.phim_irr), 0.0)
        Ga1 = self.k1 * H
        Ga2 = self.k2 * H
        Gf = self.kf * H + self.Gf0
        Gb = self.kb * H + self.Gb0
        return Ga1, Ga2, Gf, Gb

    def simulate(self, light_t, dt=1.0, substeps=4):
        """Integrate the four-state opsin under light_t (mW/mm^2, one value per step
        of width ``dt`` ms). Returns Iopto_t, same length as light_t.

        ``substeps`` sub-divides each step for forward-Euler accuracy/stability under
        bright light (Brian2 itself avoids plain euler for this model); 4 reproduces
        the bench curve to <0.05% at dt=1 ms. C1=1, O1=O2=0 initial (dark rest).
        """
        light_t = np.asarray(light_t, dtype=np.float64)
        Ga1, Ga2, Gf, Gb = self._rates(light_t)
        Gd1, Gd2, Gr0, gamma = self.Gd1, self.Gd2, self.Gr0, self.gamma
        h = dt / substeps
        C1, O1, O2 = 1.0, 0.0, 0.0
        iopto = np.empty(light_t.shape[0], dtype=np.float64)
        scale = -self.g0 * self.v_minus_E
        for s in range(light_t.shape[0]):
            a1, a2, gf, gb = Ga1[s], Ga2[s], Gf[s], Gb[s]
            for _ in range(substeps):
                C2 = 1.0 - C1 - O1 - O2
                dC1 = Gd1 * O1 + Gr0 * C2 - a1 * C1
                dO1 = a1 * C1 + gb * O2 - (Gd1 + gf) * O1
                dO2 = a2 * C2 + gf * O1 - (Gd2 + gb) * O2
                C1 += h * dC1
                O1 += h * dO1
                O2 += h * dO2
            iopto[s] = scale * (O1 + gamma * O2)
        return iopto

    def steady_state_curve(self, light_levels, dt=0.1):
        """Bench-protocol steady state (mean Iopto over the last _BENCH_TAIL_MS of a
        _BENCH_PULSE_MS constant pulse) at each light level. For validation against
        the committed calibration curve."""
        out = np.empty(len(light_levels), dtype=np.float64)
        n = int(_BENCH_PULSE_MS / dt)
        ntail = int(_BENCH_TAIL_MS / dt)
        for i, L in enumerate(light_levels):
            trace = self.simulate(np.full(n, float(L)), dt=dt, substeps=1)
            out[i] = trace[n - ntail:].mean()
        return np.abs(out)


@functools.lru_cache(maxsize=8)
def _load_calibration(path):
    with open(path) as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=8)
def derive_phim_irr(calibration_json=_DEFAULT_CALIBRATION_JSON):
    """Fit the commanded-units Hill half-saturation to the calibration curve.

    The kinetics, g0, E and v_rest are fixed by the calibration; phim_irr is the one
    free constant linking commanded light to the photon-flux Hill term. Fitting it to
    minimise relative error over the committed steady-state curve makes the surrogate
    track the plant (single source of truth: the curve JSON)."""
    cal = _load_calibration(calibration_json)
    light = np.asarray(cal["light_grid_mWmm2"], dtype=np.float64)
    ss_ref = np.asarray(cal["iopto_ss_abs"], dtype=np.float64)
    nz = ss_ref > 0
    base = dict(g0=cal["g0"], E=cal["E_rev"], v=cal["v_rest"])

    def sse(phim_irr):
        surr = OpsinSurrogate(phim_irr=phim_irr, **base)
        pred = surr.steady_state_curve(light)
        return float(np.sum(((pred[nz] - ss_ref[nz]) / ss_ref[nz]) ** 2))

    res = minimize_scalar(sse, bounds=(200.0, 8000.0), method="bounded")
    return float(res.x)


@functools.lru_cache(maxsize=8)
def load_calibrated_surrogate(calibration_json=_DEFAULT_CALIBRATION_JSON):
    """Build the surrogate from the committed calibration artifact. Cached: the
    surrogate is a pure function of the calibration, identical across sys-ID cells."""
    cal = _load_calibration(calibration_json)
    return OpsinSurrogate(
        g0=cal["g0"], E=cal["E_rev"], v=cal["v_rest"],
        phim_irr=derive_phim_irr(calibration_json),
    )
