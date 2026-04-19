"""Calibrate UnitlessChr2FourState parameters against fSNN's measured operational voltage range.

Consumes ``voperating.json`` (produced by ``measure_voperating_range.py``) and
writes ``chr2_4s_fsnn_calibration.json`` describing ``(g0, E, v0, v1)`` plus
bench-validation metrics.

Parameter derivation (see scratchpad/opsin_v3_cleanup_and_plan.md §4.3):
    window = v_high - v_low
    E  = v_low + 3.5 * window
    v0 = 2.15  * window
    v1 = 0.855 * window

``g0`` is solved by measurement, not analytically. A Brian2 long-pulse run with
``UnitlessChr2FourState(g0=1, E=E, v0=v0, v1=v1)`` at a ``v=v_median`` clamp
emits steady-state Iopto; exploiting exact linearity of Iopto in g0 gives

    g0 = target_iopto / Iopto_steady_state_at_g0_equals_1

Two bench checks then validate against the calibrated parameters:
 * target-light re-run must land within 5% of ``target_iopto``
 * saturation-light run must keep peak |Iopto| below ``peak_saturation_bound``.

Dependencies: requires ``src.opsins.UnitlessChr2FourState`` (this repo) and
``synthetic_data.synthetic_opsin_data.SynthOptoStimDataGen`` (USID repo, on
``PYTHONPATH`` via ``scripts/sbatch/run_opsin_diagnostic.sbatch``).
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import brian2.only as b2
import numpy as np

from src.opsins import UnitlessChr2FourState

# SynthOptoStimDataGen lives in the USID repo; PYTHONPATH is set by the
# sbatch wrapper (and by the tests that exercise this code) so this import
# resolves whenever calibrate_opsin_plant runs through supported entry points.
from synthetic_data.synthetic_opsin_data import SynthOptoStimDataGen

log = logging.getLogger(__name__)


# Ratios derived from published PyRhO ChR2 constants anchored to a pyramidal
# neuron's (v_rest, v_threshold) window (see opsin_justification.md §2.3).
# Applied to fSNN's measured operational window to place E/v0/v1 in fSNN's
# coordinate system while preserving biophysical ratios.
E_RATIO = 3.5
V0_RATIO = 2.15
V1_RATIO = 0.855

# Pulse timing knobs. 500 ms is well past the four-state Markov chain's
# transient (dominated by Gd2 = 0.0138/ms → τ ≈ 72 ms, plus slower O1↔O2
# transitions). Bench pulse must match calibration pulse exactly — linearity
# in g0 guarantees rel_err = 0 when both samples come from the same time
# point, but a mid-transient mismatch shows up as apparent rel_err even
# though the calibration itself is correct. Unify to the same duration.
DEFAULT_PULSE_DURATION_MS = 500.0
DEFAULT_BENCH_DURATION_MS = DEFAULT_PULSE_DURATION_MS
DEFAULT_MODEL_PERIOD_MS = 1.0
_STEADY_STATE_TAIL_SAMPLES = 10


def _dimensionless_dummy_neuron_model(v_clamp: float) -> str:
    """Brian2 equations string for a v-clamped dimensionless dummy neuron.

    Matches the form used by USID's test suite (see
    ``tests/test_chr2_4s_fsnn_opsin.py::_dimensionless_dummy_neuron``).
    The ``x/y/z`` fields satisfy cleo.Light's transmittance computation.
    """
    return (
        f"v = {v_clamp} : 1\n"
        f"            Iopto : 1\n"
        f"            x = 0*um : meter\n"
        f"            y = 0*um : meter\n"
        f"            z = 0*um : meter"
    )


def _run_constant_pulse(
    opsin: UnitlessChr2FourState,
    light_mWmm2: float,
    v_clamp: float,
    duration_ms: float,
    model_period_ms: float,
) -> np.ndarray:
    """Single cleo pulse: constant light, v-clamped dummy; returns Iopto(t)."""
    b2.start_scope()
    data_gen = SynthOptoStimDataGen(
        opto_stim_fn=lambda t: light_mWmm2,
        opsin_model=opsin,
        model_period_ms=model_period_ms,
        dummy_neuron_model=_dimensionless_dummy_neuron_model(v_clamp),
    )
    photocurrent, _ = data_gen.run_simulation(duration=duration_ms, t_unit=b2.ms)
    return np.asarray(photocurrent[1], dtype=np.float64)


def _steady_state(iopto: np.ndarray) -> float:
    if iopto.size < _STEADY_STATE_TAIL_SAMPLES:
        raise RuntimeError(
            f"Too few samples ({iopto.size}) for steady-state estimate "
            f"(needed {_STEADY_STATE_TAIL_SAMPLES})."
        )
    return float(np.mean(iopto[-_STEADY_STATE_TAIL_SAMPLES:]))


def _fv_term(v: float, E: float, v0: float, v1: float) -> float:
    """L'Hôpital-transformed voltage term from cleo's chr2_4s equations.

    ``fv_timesVminusE = v1 * (1 - exp(-(v-E)/v0))`` — cancels the v=E
    singularity by construction. Preserved byte-for-byte in UnitlessChr2FourState.
    """
    return float(v1 * (1.0 - np.exp(-(v - E) / v0)))


def _calibration_hash(core: dict) -> str:
    """Short deterministic hash over calibration output used as cache key
    and provenance tag (see plan §4.5, §4.6)."""
    blob = json.dumps(core, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def calibrate(
    voperating: dict,
    target_iopto: float = 0.1,
    target_light_mWmm2: float = 5.0,
    saturation_light_mWmm2: float = 75.0,
    pulse_duration_ms: float = DEFAULT_PULSE_DURATION_MS,
    bench_duration_ms: float = DEFAULT_BENCH_DURATION_MS,
    model_period_ms: float = DEFAULT_MODEL_PERIOD_MS,
    rel_err_tolerance: float = 0.05,
    peak_saturation_bound: float = 1.5,
) -> dict:
    """Derive (g0, E, v0, v1) from measured fSNN v-range + bench-validate."""
    v_low = float(voperating["v_low"])
    v_high = float(voperating["v_high"])
    v_median = float(voperating["v_median"])
    window = v_high - v_low
    if window <= 0:
        raise ValueError(
            f"Degenerate voperating window: v_high ({v_high}) <= v_low ({v_low})."
        )

    E = v_low + E_RATIO * window
    v0 = V0_RATIO * window
    v1 = V1_RATIO * window

    fv_at_median = _fv_term(v_median, E, v0, v1)
    if fv_at_median == 0.0:
        raise RuntimeError(
            "fv_term evaluated to 0 at v=v_median — would divide by zero when "
            "solving for g0. Check voperating window (v_median coincides with E?)."
        )

    log.info(
        f"Derived params: window={window:.4f}, E={E:.4f}, v0={v0:.4f}, v1={v1:.4f}; "
        f"fv_term(v_median={v_median:.4f}) = {fv_at_median:.6f}"
    )

    # Step 1: measure steady-state Iopto with g0=1. Exact linearity of the
    # chr2_4s current equation in g0 lets us solve for the calibrated g0
    # in closed form: Iopto_G(t) = G * Iopto_1(t) when (fphi, v) are
    # independent of g0, which they are.
    opsin_unit = UnitlessChr2FourState(g0=1.0, E=E, v0=v0, v1=v1)
    iopto_unit = _run_constant_pulse(
        opsin=opsin_unit,
        light_mWmm2=target_light_mWmm2,
        v_clamp=v_median,
        duration_ms=pulse_duration_ms,
        model_period_ms=model_period_ms,
    )
    iopto_unit_ss = _steady_state(iopto_unit)
    if iopto_unit_ss == 0.0:
        raise RuntimeError(
            "Measured Iopto steady-state with g0=1 is zero; cannot solve for g0. "
            "Opsin may not be receiving light (check fiber/transmittance)."
        )

    g0 = target_iopto / iopto_unit_ss
    # Back out fphi_ss for reporting; rho_rel=1 in the unit-fiber dummy setup.
    # Iopto = -g0 * fphi * fv_term * rho_rel  →  fphi = -Iopto / (g0 * fv_term).
    fphi_ss = -iopto_unit_ss / fv_at_median
    log.info(
        f"Unit-pulse measurement: Iopto_ss(g0=1) = {iopto_unit_ss:.6e} at "
        f"{target_light_mWmm2:g} mW/mm²; implied fphi_ss = {fphi_ss:.6f}; "
        f"solved g0 = {g0:.6e}"
    )

    # Step 2: bench validation — re-run with calibrated g0 to confirm Iopto
    # lands on target_iopto within rel_err_tolerance. Perfect linearity
    # implies exact agreement; deviation flags numerical or plumbing issues.
    opsin_cal = UnitlessChr2FourState(g0=g0, E=E, v0=v0, v1=v1)
    iopto_bench = _run_constant_pulse(
        opsin=opsin_cal,
        light_mWmm2=target_light_mWmm2,
        v_clamp=v_median,
        duration_ms=bench_duration_ms,
        model_period_ms=model_period_ms,
    )
    iopto_bench_ss = _steady_state(iopto_bench)
    rel_err = abs(iopto_bench_ss - target_iopto) / abs(target_iopto)
    log.info(
        f"Bench (calibrated g0, {target_light_mWmm2:g} mW/mm²): "
        f"Iopto_ss = {iopto_bench_ss:.6e} vs target {target_iopto:g} (rel_err = {rel_err:.2%})"
    )
    if rel_err > rel_err_tolerance:
        raise RuntimeError(
            f"Bench validation failed: rel_err {rel_err:.2%} > {rel_err_tolerance:.0%}. "
            f"Iopto_ss={iopto_bench_ss:.6e}, target={target_iopto:g}."
        )

    # Step 3: saturation check — peak |Iopto| under 75 mW/mm² must stay below
    # the absolute bound so fSNN's driving-current envelope isn't violated.
    # Need a fresh opsin instance because Brian2's start_scope() in
    # _run_constant_pulse would otherwise discard state mid-use.
    opsin_sat = UnitlessChr2FourState(g0=g0, E=E, v0=v0, v1=v1)
    iopto_sat = _run_constant_pulse(
        opsin=opsin_sat,
        light_mWmm2=saturation_light_mWmm2,
        v_clamp=v_median,
        duration_ms=bench_duration_ms,
        model_period_ms=model_period_ms,
    )
    peak_sat = float(np.max(np.abs(iopto_sat)))
    log.info(
        f"Saturation ({saturation_light_mWmm2:g} mW/mm²): peak |Iopto| = {peak_sat:.6e} "
        f"vs bound {peak_saturation_bound:g}"
    )
    if peak_sat > peak_saturation_bound:
        raise RuntimeError(
            f"Saturation check failed: peak |Iopto| {peak_sat:.6e} > bound {peak_saturation_bound:g} "
            f"at {saturation_light_mWmm2:g} mW/mm². fSNN's input envelope would be violated; "
            f"revisit v_high/v_low window or target_iopto."
        )

    # Hash the *defining* parameters only (g0, E, v0, v1 + calibration
    # target), not derived diagnostics. Ensures hash is stable w.r.t.
    # incidental changes in bench numeric output across runs.
    core = {
        "g0": g0, "E": E, "v0": v0, "v1": v1,
        "target_iopto": target_iopto,
        "target_light_mWmm2": target_light_mWmm2,
        "E_ratio": E_RATIO, "v0_ratio": V0_RATIO, "v1_ratio": V1_RATIO,
    }
    calibration_hash = _calibration_hash(core)

    result = {
        "g0": g0,
        "E": E,
        "v0": v0,
        "v1": v1,
        "window": window,
        "v_low": v_low,
        "v_high": v_high,
        "v_median": v_median,
        "target_iopto": target_iopto,
        "target_light_mWmm2": target_light_mWmm2,
        "saturation_light_mWmm2": saturation_light_mWmm2,
        "pulse_duration_ms": pulse_duration_ms,
        "bench_duration_ms": bench_duration_ms,
        "model_period_ms": model_period_ms,
        "fphi_ss_at_target": fphi_ss,
        "fv_term_at_v_median": fv_at_median,
        "bench": {
            "iopto_ss_at_target": iopto_bench_ss,
            "rel_err": rel_err,
            "rel_err_tolerance": rel_err_tolerance,
            "peak_at_saturation": peak_sat,
            "peak_saturation_bound": peak_saturation_bound,
        },
        "ratios": {"E": E_RATIO, "v0": V0_RATIO, "v1": V1_RATIO},
        "voperating_ref": {
            "vr": voperating.get("vr"),
            "vth": voperating.get("vth"),
            "cleo_prep_file": voperating.get("cleo_prep_file"),
        },
        "calibration_hash": calibration_hash,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("voperating_json", type=str,
                        help="Path to voperating.json from measure_voperating_range.")
    parser.add_argument("out_json", type=str,
                        help="Path to write calibration JSON.")
    parser.add_argument("--target-iopto", type=float, default=0.1,
                        help="Desired steady-state Iopto at target light.")
    parser.add_argument("--target-light", type=float, default=5.0,
                        help="Target light intensity in mW/mm².")
    parser.add_argument("--saturation-light", type=float, default=75.0,
                        help="Saturation check light intensity in mW/mm².")
    parser.add_argument("--duration-ms", type=float, default=DEFAULT_PULSE_DURATION_MS,
                        help="Calibration pulse duration (well past transient).")
    parser.add_argument("--bench-duration-ms", type=float, default=DEFAULT_BENCH_DURATION_MS,
                        help="Bench/saturation pulse duration.")
    parser.add_argument("--model-period-ms", type=float, default=DEFAULT_MODEL_PERIOD_MS,
                        help="Brian2 model period (Iopto sampling resolution).")
    parser.add_argument("--rel-err-tolerance", type=float, default=0.05,
                        help="Bench tolerance for |Iopto_ss - target|/target.")
    parser.add_argument("--peak-saturation-bound", type=float, default=1.5,
                        help="Absolute bound on |Iopto| at saturation-light input. "
                             "Default 1.5 matches v1 ProportionalCurrentOpsin's Iopto "
                             "at 75 mW/mm² (15× the calibration light), i.e. v1-parity. "
                             "See scratchpad/v3_opsin_deviations.md D5.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with open(args.voperating_json) as f:
        voperating = json.load(f)

    result = calibrate(
        voperating=voperating,
        target_iopto=args.target_iopto,
        target_light_mWmm2=args.target_light,
        saturation_light_mWmm2=args.saturation_light,
        pulse_duration_ms=args.duration_ms,
        bench_duration_ms=args.bench_duration_ms,
        model_period_ms=args.model_period_ms,
        rel_err_tolerance=args.rel_err_tolerance,
        peak_saturation_bound=args.peak_saturation_bound,
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    log.info(f"calibration complete → {out_path} (hash={result['calibration_hash']})")


if __name__ == "__main__":
    main()
