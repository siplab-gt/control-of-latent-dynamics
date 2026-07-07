"""sys-ID stimulation methods, selected by the ``stim_method`` wildcard.

Each method builds a sampler with a ``.step() -> (n_channels,) light array (mW/mm^2)``
called once per command step in sys_id_expts. All methods share the actuator cap
(``stim_u_max_mWmm2``) and the operating point (``stim_budget_mWmm2`` = mean light), so
they are compared at a matched operating point. Add a method = add a file + a branch here
+ a name in STIM_METHODS; sweep it via a ``sweeps:`` block on ``stim_method``.

  oed    : Fisher-information OED, differential_evolution optimizer (src.stim.oed)
  white  : legacy white-lognormal x Bernoulli, duty tuned to the budget (src.stim.white)
  chirp  : log-frequency sweep at the operating point (src.stim.chirp)
"""
from src.stim.oed import derive_optimal_design
from src.stim.chirp import ChirpSampler
from src.stim.white import WhiteLognormalSampler

STIM_METHODS = ("oed", "white", "chirp")


def build_stim_sampler(method, cond_params, base_params, n_channels, command_dt_ms, rng):
    u_max = float(cond_params.stim_u_max_mWmm2)
    budget = float(cond_params.stim_budget_mWmm2)
    if method == "oed":
        # target_iopto only sets g0, which rescales the saturation curve without
        # changing its shape, so the OED design is g0-invariant; but read the produced
        # per-ti curve (a declared sys_id_expts input) rather than a global constant so
        # the calibration source stays DAG-consistent and clean-run-safe (the global
        # path is not produced under the target_iopto wildcard).
        calib_dir = getattr(base_params, "opsin_calibration_dir", "results/opsin_calibration")
        calibration_json = (
            f"{calib_dir}/target_iopto_{cond_params.target_iopto}"
            "/chr2_4s_fsnn_saturation_curve.json"
        )
        design = derive_optimal_design(
            u_max, budget, command_dt_ms, calibration_json=calibration_json
        )
        return design.make_sampler(n_channels, rng)
    if method == "white":
        return WhiteLognormalSampler(
            n_channels, u_max, budget,
            float(base_params.typical_Irr0_mWmm2), command_dt_ms, rng)
    if method == "chirp":
        return ChirpSampler(n_channels, u_max, budget, command_dt_ms, rng)
    raise ValueError(f"unknown stim_method {method!r}; valid: {STIM_METHODS}")
