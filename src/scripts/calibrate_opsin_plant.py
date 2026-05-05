"""Calibrate UnitlessChr2FourState (Bansal-derived) parameters against fSNN's measured operational voltage range.

Consumes ``voperating.json`` (produced by ``measure_voperating_range.py``) and
writes ``chr2_4s_fsnn_calibration.json`` describing ``(g0, E)`` plus
bench-validation metrics.

Parameter derivation (Bansal-anchored):
    window = v_high - v_low
    E      = v_low + 3.5 * window     (anchored away from the operating range
                                       to keep |V−E| well-conditioned)

Bansal's driving force is linear: ``Iopto ∝ -g0 * fphi * (V − E)``. This is
simpler than the PyRhO ``fv_timesVminusE`` form (no v0/v1 saturation params)
and the v=E singularity is replaced by an exact zero (no l'Hôpital needed).

``g0`` is solved by measurement, not analytically. A Brian2 long-pulse run
with ``UnitlessChr2FourState(g0=1, E=E)`` at a ``v=v_median`` clamp emits
steady-state Iopto; exploiting exact linearity of Iopto in g0 gives

    g0 = target_iopto / Iopto_steady_state_at_g0_equals_1

Three bench checks then validate against the calibrated parameters:
 * target-light re-run must land within ``rel_err_tolerance`` of ``target_iopto``
 * target-light peak Iopto must stay below 2 × target_iopto
   (catches Bansal kinetic-shape changes that wouldn't be visible at saturation)
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


# Anchor E away from fSNN's operating range so |V−E| is well-conditioned across
# the measured window. The 3.5×window placement is preserved from the PyRhO
# calibration to minimize cross-method drift in the steady-state operating
# point — only the driving-force functional form changed (linear vs saturating).
E_RATIO = 3.5

# Pulse timing knobs. 500 ms is well past the four-state Markov chain's
# transient (dominated by Gd2 = 0.01/ms → τ ≈ 100 ms under Bansal). Bench
# pulse must match calibration pulse exactly — linearity in g0 guarantees
# rel_err = 0 when both samples come from the same time point, but a
# mid-transient mismatch shows up as apparent rel_err even though the
# calibration itself is correct. Unify to the same duration.
DEFAULT_PULSE_DURATION_MS = 500.0
DEFAULT_BENCH_DURATION_MS = DEFAULT_PULSE_DURATION_MS
DEFAULT_MODEL_PERIOD_MS = 1.0
_STEADY_STATE_TAIL_SAMPLES = 10

# Bansal's saturation light produces a smaller |Iopto| envelope than PyRhO at
# the same target_iopto, so the legacy bound (1.5) leaves orders-of-magnitude
# headroom that masks Bansal kinetic-shape regressions. Tighten to 0.3, which
# is ≈20% above the empirically observed PyRhO peak (~0.25 at 75 mW/mm² with
# target_iopto=0.05) — a Bansal recalibration that produces a much larger
# peak fails the bench check rather than silently driving fSNN out of regime.
DEFAULT_PEAK_SATURATION_BOUND = 0.3


def _dimensionless_dummy_neuron_model(v_clamp: float) -> str:
    """Brian2 equations string for a v-clamped dimensionless dummy neuron.

    Matches the form used by USID's NN training path
    (``synthetic_data.synthetic_opsin_data.SynthOptoStimDataGen``) and by
    USID's test suite (``tests/test_chr2_4s_fsnn_opsin.py::_dimensionless_dummy_neuron``).
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


def _calibration_hash(core: dict) -> str:
    """Short deterministic hash over calibration output used as cache key
    and provenance tag."""
    blob = json.dumps(core, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def calibrate(
    voperating: dict,
    target_iopto: float,
    target_light_mWmm2: float = 5.0,
    saturation_light_mWmm2: float = 75.0,
    pulse_duration_ms: float = DEFAULT_PULSE_DURATION_MS,
    bench_duration_ms: float = DEFAULT_BENCH_DURATION_MS,
    model_period_ms: float = DEFAULT_MODEL_PERIOD_MS,
    rel_err_tolerance: float = 0.05,
    peak_saturation_bound: float = DEFAULT_PEAK_SATURATION_BOUND,
) -> dict:
    """Derive (g0, E) from measured fSNN v-range + bench-validate (Bansal)."""
    v_low = float(voperating["v_low"])
    v_high = float(voperating["v_high"])
    v_median = float(voperating["v_median"])
    window = v_high - v_low
    if window <= 0:
        raise ValueError(
            f"Degenerate voperating window: v_high ({v_high}) <= v_low ({v_low})."
        )

    E = v_low + E_RATIO * window
    v_minus_E = v_median - E
    if v_minus_E == 0.0:
        raise RuntimeError(
            "(v_median - E) evaluated to 0 — would divide by zero when "
            "solving for g0. Check voperating window (v_median coincides with E?)."
        )

    log.info(
        f"Derived params (Bansal): window={window:.4f}, E={E:.4f}; "
        f"(v_median={v_median:.4f} − E) = {v_minus_E:.6f}"
    )

    # Step 1: measure steady-state Iopto with g0=1. Exact linearity of the
    # Bansal current equation in g0 lets us solve for the calibrated g0
    # in closed form: Iopto_G(t) = G * Iopto_1(t) when (fphi, V−E) are
    # independent of g0, which they are.
    opsin_unit = UnitlessChr2FourState(g0=1.0, E=E)
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
    # Iopto = -g0 * fphi * (V−E) * rho_rel  →  fphi = -Iopto / (g0 * (V−E)).
    # (At g0=1: fphi = -iopto_unit_ss / v_minus_E.)
    fphi_ss = -iopto_unit_ss / v_minus_E
    log.info(
        f"Unit-pulse measurement: Iopto_ss(g0=1) = {iopto_unit_ss:.6e} at "
        f"{target_light_mWmm2:g} mW/mm²; implied fphi_ss = {fphi_ss:.6f}; "
        f"solved g0 = {g0:.6e}"
    )

    # Step 2: bench validation — re-run with calibrated g0 to confirm Iopto
    # lands on target_iopto within rel_err_tolerance. Perfect linearity
    # implies exact agreement; deviation flags numerical or plumbing issues.
    # Also captures peak Iopto at target light to gate transient overshoot.
    opsin_cal = UnitlessChr2FourState(g0=g0, E=E)
    iopto_bench = _run_constant_pulse(
        opsin=opsin_cal,
        light_mWmm2=target_light_mWmm2,
        v_clamp=v_median,
        duration_ms=bench_duration_ms,
        model_period_ms=model_period_ms,
    )
    iopto_bench_ss = _steady_state(iopto_bench)
    rel_err = abs(iopto_bench_ss - target_iopto) / abs(target_iopto)
    peak_at_target = float(np.max(np.abs(iopto_bench)))
    log.info(
        f"Bench (calibrated g0, {target_light_mWmm2:g} mW/mm²): "
        f"Iopto_ss = {iopto_bench_ss:.6e} vs target {target_iopto:g} (rel_err = {rel_err:.2%}); "
        f"peak |Iopto| = {peak_at_target:.6e}"
    )
    if rel_err > rel_err_tolerance:
        raise RuntimeError(
            f"Bench validation failed: rel_err {rel_err:.2%} > {rel_err_tolerance:.0%}. "
            f"Iopto_ss={iopto_bench_ss:.6e}, target={target_iopto:g}."
        )

    # Pre-saturation peak gate: transient overshoot at target light must stay
    # under 2× steady-state. Catches Bansal kinetic-shape changes (e.g., a
    # faster O1 transient) that wouldn't be visible in the saturation-only
    # check and could push fSNN spike rates outside the working regime.
    peak_target_bound = 2.0 * abs(target_iopto)
    if peak_at_target > peak_target_bound:
        raise RuntimeError(
            f"Target-light peak gate failed: peak |Iopto| {peak_at_target:.6e} "
            f"> 2×target {peak_target_bound:.6e} at {target_light_mWmm2:g} mW/mm². "
            f"Bansal kinetics produced a transient overshoot that may drive fSNN "
            f"out of its working regime."
        )

    # Step 3: saturation check — peak |Iopto| under 75 mW/mm² must stay below
    # the absolute bound so fSNN's driving-current envelope isn't violated.
    # Need a fresh opsin instance because Brian2's start_scope() in
    # _run_constant_pulse would otherwise discard state mid-use.
    opsin_sat = UnitlessChr2FourState(g0=g0, E=E)
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

    # Hash the *defining* parameters only (g0, E + calibration target),
    # not derived diagnostics. Ensures hash is stable w.r.t. incidental
    # changes in bench numeric output across runs.
    core = {
        "g0": g0, "E": E,
        "target_iopto": target_iopto,
        "target_light_mWmm2": target_light_mWmm2,
        "E_ratio": E_RATIO,
        "variant": "bansal",
    }
    calibration_hash = _calibration_hash(core)

    result = {
        "g0": g0,
        "E": E,
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
        "v_minus_E_at_v_median": v_minus_E,
        "bench": {
            "iopto_ss_at_target": iopto_bench_ss,
            "rel_err": rel_err,
            "rel_err_tolerance": rel_err_tolerance,
            "peak_at_target": peak_at_target,
            "peak_target_bound": peak_target_bound,
            "peak_at_saturation": peak_sat,
            "peak_saturation_bound": peak_saturation_bound,
        },
        "ratios": {"E": E_RATIO},
        "voperating_ref": {
            "vr": voperating.get("vr"),
            "vth": voperating.get("vth"),
            "cleo_prep_file": voperating.get("cleo_prep_file"),
        },
        "calibration_hash": calibration_hash,
        "variant": "bansal",
        "cleo_class": {
            "subclass": "src.opsins.UnitlessChr2FourState",
            "parent": "cleo.opto.opsins.BansalFourStateOpsin",
            "variant": "bansal",
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("voperating_json", type=str,
                        help="Path to voperating.json from measure_voperating_range.")
    parser.add_argument("out_json", type=str,
                        help="Path to write calibration JSON.")
    parser.add_argument("--target-iopto", type=float, required=True,
                        help="Desired steady-state Iopto at target light. REQUIRED — "
                             "no default because the value has shifted repeatedly "
                             "across calibration generations and is likely to shift "
                             "again as fSNN-stability calibration tightens. "
                             "Snakemake rule `calibrate_opsin_plant` passes this from "
                             "workflow config.")
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
    parser.add_argument("--peak-saturation-bound", type=float,
                        default=DEFAULT_PEAK_SATURATION_BOUND,
                        help="Absolute bound on |Iopto| at saturation-light input. "
                             "Default 0.3 (≈20%% margin above the empirically observed "
                             "PyRhO peak ~0.25); Bansal-anchored calibration produces a "
                             "smaller envelope, so this gate catches kinetic-shape "
                             "regressions that the legacy 1.5 bound would mask.")
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
    log.info(f"calibration complete → {out_path} (hash={result['calibration_hash']}, variant=bansal)")


if __name__ == "__main__":
    main()
