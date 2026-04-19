"""Measure fSNN's operational voltage range under representative sys-ID stimulation.

Builds a minimal CleoPrep inline (forcing ``proportional_current`` plant
regardless of the configured ``opsin_plant_class``) so this measurement can
run before the v3 plant is calibrated. The main ``cleo_prep`` rule depends
on ``chr2_4s_fsnn_calibration.json`` under the v3 plant; calibration in
turn depends on this measurement; so voperating must not route through
the main rule or we deadlock.

Scientific note: the v-distribution is dominated by fSNN's intrinsic
dynamics, so measuring it under ``proportional_current`` is an acceptable
approximation of what the ``chr2_4s_fsnn`` plant will see at the calibrated
operating point (both plants deliver ~0.1 Iopto at 5 mW/mm²; they diverge
at high-drive transients, which don't dominate the v-distribution).
See scratchpad/v3_opsin_deviations.md D1.

Outputs consumed by v3 opsin calibration (see
scratchpad/opsin_v3_cleanup_and_plan.md §4.1a, §4.3):
  - v_low  = 5th percentile  → calibration window lower bound
  - v_high = 95th percentile → calibration window upper bound
  - v_median = 50th percentile → diagnostic only
  - v_stdev_normalized = population std / (v_high - v_low)

Invoked via Snakemake's ``script:`` directive — see rule
``measure_voperating_range`` in ``workflow/Snakefile``.
"""
import copy
import json
import logging
from pathlib import Path

import brian2.only as b2
import cleo
import matplotlib.pyplot as plt
import numpy as np

from src.fsnn import fSNN
from src.helpers import (
    PassthroughIOProcessor,  # noqa: F401 — dill needs it in-scope for un-pickling
    preprocess_params,
    setup_logging,
)
from src.scripts.cleo_prep import CleoPrep

log = logging.getLogger(__name__)


class LognormalLightProcessor(cleo.LatencyIOProcessor):
    """IO processor that emits a sys-ID-style lognormal light vector each step.

    Matches the inline stim logic in sys_id_expts.py — same distribution,
    same clip. Refactored here so the measurement script can reuse it
    without the trial/spike-collection concerns of SysIDProcessor.
    """

    def __init__(self, sample_period, k_channels, u_max_mWmm2, typical_Irr0_mWmm2, rng):
        super().__init__(sample_period=sample_period)
        self._k = int(k_channels)
        self._u_max = float(u_max_mWmm2)
        self._u_logmean = float(np.log(typical_Irr0_mWmm2))
        self._u_logstd = float(np.log(u_max_mWmm2) / 3)
        self._rng = rng

    def process(self, state_dict, t_samp):
        value = (
            self._rng.lognormal(self._u_logmean, self._u_logstd, self._k).clip(0, self._u_max)
            * b2.mwatt / b2.mm2
        )
        return {"light array": value}, t_samp


def _zero_task_inputs(fsnn) -> None:
    fsnn.mode = "test"
    fsnn.fin_ta = b2.TimedArray(np.zeros((1, fsnn.n_in)), fsnn.dt)
    fsnn.fout_ta = b2.TimedArray(np.zeros((1, fsnn.n_out)), fsnn.dt)
    fsnn.ytilde_ta = b2.TimedArray(np.zeros((1, fsnn.n_factors)), fsnn.dt)


def measure_voperating_range(
    fsnn_file: str,
    cond_params,
    base_params,
    out_path: Path,
    settle_ms: float = 200.0,
    measure_ms: float = 1000.0,
    u_max_mWmm2: float | None = None,
    typical_Irr0_mWmm2: float | None = None,
    subsample_factor: int | None = None,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)

    # Force proportional_current regardless of configured plant class.
    # CleoPrep.build then takes the proportional_current branch and does
    # not consult the chr2_4s_fsnn calibration file, which cannot exist
    # at this pre-calibration point in the DAG.
    base_params = copy.copy(base_params)
    base_params["opsin_plant_class"] = "proportional_current"

    fsnn = fSNN.load(fsnn_file)
    log.info(f"Loaded fSNN from {fsnn_file}")
    cleo_prep = CleoPrep.build(fsnn, cond_params, base_params)
    log.info("Built CleoPrep inline with proportional_current plant")

    # Resolve drive params: explicit kwargs > base_params defaults.
    if u_max_mWmm2 is None:
        u_max_mWmm2 = float(base_params.get("u_max_mWmm2", 75.0))
    if typical_Irr0_mWmm2 is None:
        typical_Irr0_mWmm2 = float(base_params.get("typical_Irr0_mWmm2", 5.0))
    if subsample_factor is None:
        subsample_factor = int(base_params.get("subsample_factor", 10))

    _zero_task_inputs(fsnn)
    cleo_prep.reset()

    sample_period = fsnn.dt * subsample_factor
    k_channels = cleo_prep.light.n_groups
    cleo_prep.sim.set_io_processor(LognormalLightProcessor(
        sample_period=sample_period,
        k_channels=k_channels,
        u_max_mWmm2=u_max_mWmm2,
        typical_Irr0_mWmm2=typical_Irr0_mWmm2,
        rng=rng,
    ))

    v_mon = b2.StateMonitor(fsnn.ng_main, "v", record=True, name="voperating_v_mon")
    fsnn.net.add(v_mon)

    ns = fsnn._get_namespace(t_ta_start=0 * b2.ms)
    log.info(f"Running settle ({settle_ms} ms)...")
    cleo_prep.sim.run(settle_ms * b2.ms, namespace=ns)
    n_settle_samples = int(round(settle_ms * b2.ms / fsnn.dt))
    log.info(
        f"Running measurement ({measure_ms} ms) with {k_channels}-channel lognormal drive "
        f"(mu=ln{typical_Irr0_mWmm2:g}, sigma=ln{u_max_mWmm2:g}/3, cap={u_max_mWmm2:g} mW/mm²)..."
    )
    cleo_prep.sim.run(measure_ms * b2.ms, namespace=ns)

    # (N_neurons, T) → pool for percentiles.
    v_samples = np.asarray(v_mon.v)[:, n_settle_samples:]
    v_pooled = v_samples.ravel()

    v_low = float(np.percentile(v_pooled, 5))
    v_high = float(np.percentile(v_pooled, 95))
    v_median = float(np.percentile(v_pooled, 50))
    window = v_high - v_low
    pop_std = float(v_samples.std())
    v_stdev_normalized = float(pop_std / window) if window > 0 else float("nan")

    # fSNN's spike threshold / reset voltage — exact constants from fSNN config.
    # Emitted for provenance; the v-histogram's two dominant peaks sit near
    # these values (slow ramp-to-threshold + reset snap-back).
    vr = float(fsnn._base_params.vr)
    vth = float(fsnn._base_params.vth)

    peak_window = 0.5
    fraction_near_vr = float(np.mean(np.abs(v_pooled - vr) < peak_window))
    fraction_near_vth = float(np.mean(np.abs(v_pooled - vth) < peak_window))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "v_low": v_low,
        "v_high": v_high,
        "v_median": v_median,
        "window": window,
        "v_stdev_normalized": v_stdev_normalized,
        "vr": vr,
        "vth": vth,
        "fraction_near_vr": fraction_near_vr,
        "fraction_near_vth": fraction_near_vth,
        "peak_mass_window": peak_window,
        "n_neurons": int(v_samples.shape[0]),
        "n_samples_per_neuron": int(v_samples.shape[1]),
        "measure_ms": measure_ms,
        "settle_ms": settle_ms,
        "u_max_mWmm2": u_max_mWmm2,
        "typical_Irr0_mWmm2": typical_Irr0_mWmm2,
        "k_channels": k_channels,
        "subsample_factor": subsample_factor,
        "fsnn_file": str(fsnn_file),
        "plant_class_override": "proportional_current",
        "seed": seed,
    }
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    log.info(
        f"v: p5={v_low:.4f}, p50={v_median:.4f}, p95={v_high:.4f} "
        f"(window={window:.4f}, norm_std={v_stdev_normalized:.3f}); "
        f"peak mass: near vr={fraction_near_vr:.3f}, near vth={fraction_near_vth:.3f} → {out_path}"
    )

    # Histogram — critical for unimodality inspection per §4.1a.
    hist_path = out_path.with_suffix(".png")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(v_pooled, bins=120, density=True, alpha=0.75, color="steelblue",
            label=f"v pooled (N×T = {len(v_pooled):,})")
    for p, lbl, c in [(v_low, "p5", "C1"), (v_median, "p50", "C2"), (v_high, "p95", "C3")]:
        ax.axvline(p, linestyle="--", color=c, label=f"{lbl}={p:.3f}")
    ax.set_xlabel("v (dimensionless)")
    ax.set_ylabel("density")
    ax.set_title(
        f"fSNN v(t) under sys-ID-style lognormal light "
        f"(μ=ln{typical_Irr0_mWmm2:g}, σ=ln{u_max_mWmm2:g}/3, cap={u_max_mWmm2:g} mW/mm², "
        f"k={k_channels}, plant=proportional_current)"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    log.info(f"histogram → {hist_path}")

    return result


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    log_fh = setup_logging(snakemake.config, snakemake.log[0])
    measure_voperating_range(
        fsnn_file=snakemake.input.fsnn,
        cond_params=preprocess_params(snakemake.params.cond_params),
        base_params=preprocess_params(snakemake.params.base_params),
        out_path=Path(snakemake.output.json),
    )
