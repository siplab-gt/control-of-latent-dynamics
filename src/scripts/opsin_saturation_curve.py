"""Characterize the calibrated opsin's steady-state saturation curve.

Reads a `chr2_4s_fsnn_calibration.json` (the output of
`calibrate_opsin_plant`), rebuilds the exact calibrated opsin, sweeps
constant light over the commandable range, and writes the steady-state
response curve plus scalar linearity/saturation indices to a JSON.

This is a leaf diagnostic (analogous to `chr2_stability_check`): nothing
upstream depends on it, and `analyze_ctrl_data` reads its output to (a)
report the plant's static linearity as provenance and (b) evaluate the
controller's realized commanded-light distribution against this curve
(the operating-regime linearity metrics `ctrl_opsin_*`).

One source of truth: the constant-pulse bench and opsin construction are
reused verbatim from `calibrate_opsin_plant` (`_run_constant_pulse`,
`_steady_state`, `_default_phim`), the same code that solved g0 and that
the offline phim linearity scan used.

linearity index lin[a,b] = ss(b) / ((b/a) * ss(a)): 1.0 = perfectly
linear through the origin over [a,b], < 1.0 = saturating. lin[5,75] uses
the calibration target light (5 mW) and u_max (75 mW); lin[5,20] uses the
operating band the controller mostly commands (<= 20 mW).
"""
import json
import sys

import numpy as np

from src.opsins import UnitlessChr2FourState
from src.scripts.calibrate_opsin_plant import (
    DEFAULT_MODEL_PERIOD_MS,
    DEFAULT_PULSE_DURATION_MS,
    _default_phim,
    _run_constant_pulse,
    _steady_state,
)

OPERATING_BAND_HI_MWMM2 = 20.0
PLATEAU_TAIL_MWMM2 = (120.0, 200.0)


def build_light_grid(target_light, saturation_light):
    """Light levels (mW/mm2) spanning 0 -> u_max plus a plateau tail for the
    asymptote estimate. Anchored exactly on target_light, the operating-band
    edge, and saturation_light so metric interpolations land on grid points."""
    grid = np.concatenate([
        np.linspace(0.0, target_light, 5),
        np.linspace(target_light, OPERATING_BAND_HI_MWMM2, 5)[1:],
        np.linspace(OPERATING_BAND_HI_MWMM2, saturation_light, 5)[1:],
        np.array(PLATEAU_TAIL_MWMM2),
    ])
    return np.unique(np.round(grid, 4))


def build_opsin(cal):
    """Rebuild the calibrated opsin EXACTLY as calibrate() built its final one:
    UnitlessChr2FourState(g0=g0, E=E, phim=phim_scale*default_phim)."""
    return UnitlessChr2FourState(
        g0=float(cal["g0"]),
        E=float(cal["E"]),
        phim=float(cal["phim_scale"]) * _default_phim(),
    )


def sweep_steady_state(cal, light_grid):
    """Constant-light steady-state |Iopto| at each light level (fresh opsin per
    level; _run_constant_pulse calls start_scope and discards state)."""
    v_clamp = float(cal["v_median"])
    fiber = float(cal["fiber_distance_um"])
    ss = np.empty_like(light_grid, dtype=np.float64)
    for i, light in enumerate(light_grid):
        trace = _run_constant_pulse(
            opsin=build_opsin(cal),
            light_mWmm2=float(light),
            v_clamp=v_clamp,
            duration_ms=DEFAULT_PULSE_DURATION_MS,
            model_period_ms=DEFAULT_MODEL_PERIOD_MS,
            fiber_distance_um=fiber,
        )
        ss[i] = abs(_steady_state(trace))
    return ss


def compute_indices(light, ss, target_light, saturation_light):
    ss_at_target = float(np.interp(target_light, light, ss))
    ss_at_op_hi = float(np.interp(OPERATING_BAND_HI_MWMM2, light, ss))
    ss_at_sat = float(np.interp(saturation_light, light, ss))
    asymptote = float(ss[-1])
    ratio_75_5 = saturation_light / target_light
    ratio_20_5 = OPERATING_BAND_HI_MWMM2 / target_light
    lin_5_75 = ss_at_sat / (ratio_75_5 * ss_at_target) if ss_at_target else float("nan")
    lin_5_20 = ss_at_op_hi / (ratio_20_5 * ss_at_target) if ss_at_target else float("nan")
    i_half = (
        float(np.interp(0.5 * asymptote, ss, light))
        if asymptote > ss[0]
        else float("nan")
    )
    grad = np.gradient(ss, light)
    slope_origin = float(np.interp(light[1], light, grad))
    slope_sat = float(np.interp(saturation_light, light, grad))
    marginal_gain_at_umax = slope_sat / slope_origin if slope_origin else float("nan")
    return {
        "linearity_index_5_to_75": lin_5_75,
        "linearity_index_5_to_20": lin_5_20,
        "i_half_light_mWmm2": i_half,
        "marginal_gain_at_umax": marginal_gain_at_umax,
        "iopto_ss_at_target": ss_at_target,
        "iopto_ss_at_saturation": ss_at_sat,
        "asymptotic_max_ss": asymptote,
        "frac_of_asymptote_at_saturation": ss_at_sat / asymptote if asymptote else float("nan"),
        "slope_ss_near_origin": slope_origin,
        "slope_ss_at_saturation": slope_sat,
    }


def opsin_saturation_curve(calibration_file, out_json):
    with open(calibration_file) as f:
        cal = json.load(f)
    target_light = float(cal["target_light_mWmm2"])
    saturation_light = float(cal["saturation_light_mWmm2"])

    light_grid = build_light_grid(target_light, saturation_light)
    ss = sweep_steady_state(cal, light_grid)
    indices = compute_indices(light_grid, ss, target_light, saturation_light)

    out = {
        "description": (
            "Calibrated UnitlessChr2FourState steady-state |Iopto| vs constant "
            "light. Curve + linearity indices for the operating-regime control "
            "metric. Bench reused from calibrate_opsin_plant."
        ),
        "calibration_hash": cal.get("calibration_hash"),
        "phim_scale": float(cal["phim_scale"]),
        "g0": float(cal["g0"]),
        "target_light_mWmm2": target_light,
        "saturation_light_mWmm2": saturation_light,
        "operating_band_hi_mWmm2": OPERATING_BAND_HI_MWMM2,
        "light_grid_mWmm2": light_grid.tolist(),
        "iopto_ss_abs": ss.tolist(),
        **indices,
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"opsin saturation curve: lin[5,75]={indices['linearity_index_5_to_75']:.3f} "
        f"lin[5,20]={indices['linearity_index_5_to_20']:.3f} "
        f"I_half={indices['i_half_light_mWmm2']:.1f}mW "
        f"-> {out_json}"
    )


if __name__ == "__main__":
    opsin_saturation_curve(sys.argv[1], sys.argv[2])
