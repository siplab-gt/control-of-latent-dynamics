"""chr2 stability check for the fSNN.

Compares chr2-attached fSNN under zero-light + task-input-driven trials
(from `natural_electrode_expts` → `natural_electrode_data.pkl.gz`,
ExptData) against the bare fSNN baseline (from `prep_fsnn_test` →
`fsnn_test_data.pkl.gz`, dict-of-monitor-states). chr2 carries the opsin
device attached to every neuron in `ng_main` but at 0 mW/mm²; any
deviation from bare is a chr2-attachment artifact.

Metrics (config defaults in `chr2_stability_check:` block):

  1. **iopto_max_abs**, **iopto_mean_abs** — absolute photocurrent
     under zero light. Should be ≈0 by physics. Direct verification of
     the "0-light is 0" invariant; if violated, the rest of the
     comparison is moot.
  2. **factor_mag_ratio** = mean(|X_chr2|) / mean(|y_bare|).
     Magnitude proxy for firing-rate inflation (X is the post-`y_`
     factor state, same as bare's `y_mon["y_"]`). Replaces the prior
     calibration's 5-17% rate-inflation metric — at 0 light the
     opto current should be ~0, so the proxy should be ~1.0.
  3. **z_mean_abs_ratio** = mean(|Z_chr2|) / mean(|Z_bare|).
     Behavioral-output magnitude shift.
  4. **z_max_ratio** = max(|Z_chr2|) / max(|Z_bare_subsampled|).
     Catches saturation / divergence the mean misses. Bare is
     subsampled to chr2's grid (`Z_bare[::subsample_factor]`) before
     computing max so extreme-value statistics aren't biased by the
     10× sample-count difference between the two recording grids.
     The histogram in the plot uses the same subsampled view.
  5. **n_nonfinite_chr2**, **n_nonfinite_bare** — NaN/inf counts in
     X/Y/Z and y_/z respectively. Hard pathology flags; any non-zero
     count fails the verdict.

Verdict `chr2_stable` is True iff all metrics pass.

Output:
  * `chr2_stability.csv` — single-row csv with metrics + verdict +
    threshold values. Merged into `summary.csv` via
    `analyze_ctrl_data` so flagged cells surface in the aggregator.
  * `chr2_stability_plot.webp` — Z trajectory overlay + histograms.

Pinned to setting=usid_stim_rec (rule level): chr2 attachment is
identical across stim_rec settings at fixed (seed, k, m) — the
cleo_prep module is aliased (see `src/scripts/cleo_prep.py`'s
`cleo_prep_module_for` dict). One stability artifact per (seed, k, m)
serves all three settings' downstream analyzers.

CAVEAT — chr2_4s vacuity: cleo's `UnitlessChr2FourState` has near-zero
resting Iopto at v_rest by design (the open-state H0 is ~0 with no
photon flux). For this opsin family the iopto sanity check should be
trivially satisfied; the rate / Z metrics here primarily exercise
Brian2 device-coupling overhead and synapse-summation arithmetic. A
PASS verdict for chr2_4s_fsnn does NOT generalize to opsins with
non-zero resting current — those need their own sanity bound.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.helpers import (
    load_pkl_gz,
    preprocess_params,
    setup_debugger,
    setup_logging,
)
from src.scripts.sys_id_expts import ExptData

log = logging.getLogger(__name__)


def _safe_ratio(num, den, eps=1e-12):
    return float(num) / float(den) if abs(den) > eps else float("nan")


def chr2_stability_check(
    cond_params,
    base_params,
    natural_data_file,
    fsnn_test_data_file,
    out_csv,
    out_plot_webp,
):
    chr2 = ExptData.load(natural_data_file)
    bare = load_pkl_gz(fsnn_test_data_file)

    # bare fSNN's Z and y_ live in z_mon["z"] and y_mon["y_"] respectively
    # (per analyze_fsnn_data.py:13-15). Skip first sample to mirror the
    # "network needs a step to update y" convention used there.
    Z_bare = np.asarray(bare["z_mon"]["z"][1:])
    y_bare = np.asarray(bare["y_mon"]["y_"][1:])
    Z_chr2 = np.asarray(chr2.Z)
    X_chr2 = np.asarray(chr2.X)  # = cleo_prep.fsnn.y_mon.y_[:, ::subsample_factor].T

    # Subsample bare to chr2's time grid (chr2 is at fSNN_dt × subsample_factor;
    # bare is at fSNN_dt). This makes max-statistics and histograms fair —
    # without it, bare's max is biased upward purely from having ~10× more
    # extreme-value draws per unit wall time. Mean-statistics are
    # sample-rate-invariant for stationary signals so don't need this, but
    # we use the subsampled view uniformly for simplicity.
    subsample_factor = int(base_params.subsample_factor)
    Z_bare_sub = Z_bare[::subsample_factor]
    y_bare_sub = y_bare[::subsample_factor]

    # ---- Z stats (bare subsampled to chr2's grid for fair max + histogram) ----
    Z_chr2_finite = Z_chr2[np.isfinite(Z_chr2)]
    Z_bare_finite = Z_bare_sub[np.isfinite(Z_bare_sub)]
    z_mean_abs_chr2 = float(np.mean(np.abs(Z_chr2_finite))) if Z_chr2_finite.size else float("nan")
    z_mean_abs_bare = float(np.mean(np.abs(Z_bare_finite))) if Z_bare_finite.size else float("nan")
    z_max_abs_chr2 = float(np.max(np.abs(Z_chr2_finite))) if Z_chr2_finite.size else float("nan")
    z_max_abs_bare = float(np.max(np.abs(Z_bare_finite))) if Z_bare_finite.size else float("nan")
    z_mean_abs_ratio = _safe_ratio(z_mean_abs_chr2, z_mean_abs_bare)
    z_max_ratio = _safe_ratio(z_max_abs_chr2, z_max_abs_bare)

    # ---- Factor magnitude (rate-inflation proxy) ----
    # Both X_chr2 and y_bare are the same `y_` factor state from `ng_y`.
    # At zero light the proxy should sit near 1.0 because there's no opto
    # current to inflate the post-`y_` state. Use the subsampled bare so
    # the comparison is consistent with the Z block above.
    X_chr2_finite = X_chr2[np.isfinite(X_chr2)]
    y_bare_finite = y_bare_sub[np.isfinite(y_bare_sub)]
    factor_mean_abs_chr2 = float(np.mean(np.abs(X_chr2_finite))) if X_chr2_finite.size else float("nan")
    factor_mean_abs_bare = float(np.mean(np.abs(y_bare_finite))) if y_bare_finite.size else float("nan")
    factor_mag_ratio = _safe_ratio(factor_mean_abs_chr2, factor_mean_abs_bare)

    # ---- Iopto sanity (the load-bearing 0-light invariant) ----
    # chr2.Iopto_per_neuron may be None if upstream didn't record it; for
    # chr2_4s_fsnn the StateMonitor in sys_id_expts always populates it.
    if chr2.Iopto_per_neuron is not None:
        iopto = np.asarray(chr2.Iopto_per_neuron)
        iopto_finite = iopto[np.isfinite(iopto)]
        iopto_max_abs = float(np.max(np.abs(iopto_finite))) if iopto_finite.size else float("nan")
        iopto_mean_abs = float(np.mean(np.abs(iopto_finite))) if iopto_finite.size else float("nan")
    else:
        iopto_max_abs = float("nan")
        iopto_mean_abs = float("nan")

    # ---- Pathology counts (chr2 + bare both — silent NaN passthrough on
    # the bare side would otherwise show up as a chr2 problem). ----
    n_nonfinite_chr2 = int(
        np.sum(~np.isfinite(chr2.X))
        + np.sum(~np.isfinite(chr2.Y))
        + np.sum(~np.isfinite(chr2.Z))
    )
    # NaN counts are over the FULL bare arrays (not subsampled) so we
    # don't miss pathology that lives at the inter-sample timesteps.
    n_nonfinite_bare = int(
        np.sum(~np.isfinite(y_bare))
        + np.sum(~np.isfinite(Z_bare))
    )

    # Thresholds via step_cfg("chr2_stability_check") merged dict.
    z_mean_ratio_min = float(base_params.z_mean_ratio_min)
    z_mean_ratio_max = float(base_params.z_mean_ratio_max)
    z_max_ratio_max = float(base_params.z_max_ratio_max)
    factor_mag_ratio_min = float(base_params.factor_mag_ratio_min)
    factor_mag_ratio_max = float(base_params.factor_mag_ratio_max)
    iopto_max_abs_threshold = float(base_params.iopto_max_abs_threshold)

    chr2_stable = bool(
        n_nonfinite_chr2 == 0
        and n_nonfinite_bare == 0
        and np.isfinite(iopto_max_abs)
        and iopto_max_abs <= iopto_max_abs_threshold
        and np.isfinite(factor_mag_ratio)
        and factor_mag_ratio_min <= factor_mag_ratio <= factor_mag_ratio_max
        and np.isfinite(z_mean_abs_ratio)
        and z_mean_ratio_min <= z_mean_abs_ratio <= z_mean_ratio_max
        and np.isfinite(z_max_ratio)
        and z_max_ratio <= z_max_ratio_max
    )

    row = {
        "chr2_iopto_max_abs": iopto_max_abs,
        "chr2_iopto_mean_abs": iopto_mean_abs,
        "chr2_factor_mean_abs_chr2": factor_mean_abs_chr2,
        "chr2_factor_mean_abs_bare": factor_mean_abs_bare,
        "chr2_factor_mag_ratio": factor_mag_ratio,
        "chr2_z_mean_abs_chr2": z_mean_abs_chr2,
        "chr2_z_mean_abs_bare": z_mean_abs_bare,
        "chr2_z_mean_abs_ratio": z_mean_abs_ratio,
        "chr2_z_max_abs_chr2": z_max_abs_chr2,
        "chr2_z_max_abs_bare": z_max_abs_bare,
        "chr2_z_max_ratio": z_max_ratio,
        "chr2_n_nonfinite_chr2": n_nonfinite_chr2,
        "chr2_n_nonfinite_bare": n_nonfinite_bare,
        "chr2_stable": chr2_stable,
        "chr2_threshold_iopto_max_abs": iopto_max_abs_threshold,
        "chr2_threshold_factor_mag_ratio_min": factor_mag_ratio_min,
        "chr2_threshold_factor_mag_ratio_max": factor_mag_ratio_max,
        "chr2_threshold_z_mean_ratio_min": z_mean_ratio_min,
        "chr2_threshold_z_mean_ratio_max": z_mean_ratio_max,
        "chr2_threshold_z_max_ratio_max": z_max_ratio_max,
    }
    log.info(
        f"chr2 stability: stable={chr2_stable} "
        f"iopto_max={iopto_max_abs:.4e} "
        f"factor_mag_ratio={factor_mag_ratio:.3f} "
        f"z_mean_ratio={z_mean_abs_ratio:.3f} "
        f"z_max_ratio={z_max_ratio:.3f} "
        f"n_nonfinite_chr2={n_nonfinite_chr2} n_nonfinite_bare={n_nonfinite_bare}"
    )
    pd.DataFrame([row]).to_csv(out_csv, index=False)

    # Use the bare-subsampled view for both trajectory and histogram so
    # the visual reflects the same time grid as chr2.
    n_z = Z_chr2.shape[1]
    fig, axes = plt.subplots(2, n_z, figsize=(4 * n_z, 6), squeeze=False)
    for d in range(n_z):
        # Top row: per-dim trajectory overlay (further plot-step for density).
        T_chr2 = Z_chr2.shape[0]
        T_bare_sub = Z_bare_sub.shape[0]
        chr2_step = max(1, T_chr2 // 2000)
        bare_step = max(1, T_bare_sub // 2000)
        axes[0, d].plot(
            np.arange(0, T_bare_sub, bare_step),
            Z_bare_sub[::bare_step, d],
            label=f"bare fSNN (subsampled ÷{subsample_factor})", alpha=0.6, lw=0.5,
        )
        axes[0, d].plot(
            np.arange(0, T_chr2, chr2_step),
            Z_chr2[::chr2_step, d],
            label="chr2-attached, 0-light", alpha=0.6, lw=0.5,
        )
        axes[0, d].set_title(f"Z dim {d} trajectory")
        axes[0, d].set_xlabel("sample (chr2 grid)")
        axes[0, d].legend(fontsize=8)

        # Bottom row: Z value histograms — same subsampled grid for both.
        finite_chr2 = Z_chr2[:, d][np.isfinite(Z_chr2[:, d])]
        finite_bare = Z_bare_sub[:, d][np.isfinite(Z_bare_sub[:, d])]
        axes[1, d].hist(finite_bare, bins=50, alpha=0.5, label="bare fSNN", density=True)
        axes[1, d].hist(finite_chr2, bins=50, alpha=0.5, label="chr2 0-light", density=True)
        axes[1, d].set_title(f"Z dim {d} distribution")
        axes[1, d].set_xlabel("Z value")
        axes[1, d].legend(fontsize=8)

    fig.suptitle(
        f"chr2 stability — seed={cond_params['seed']} k={cond_params['k']} "
        f"m={cond_params['m']} | stable={chr2_stable}"
    )
    fig.tight_layout()
    fig.savefig(out_plot_webp, dpi=120)
    plt.close(fig)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_logging(snakemake.config, snakemake.log[0])
    setup_debugger(snakemake.config["debug"])

    chr2_stability_check(
        preprocess_params(snakemake.wildcards),
        preprocess_params(snakemake.params),
        snakemake.input.natural_data,
        snakemake.input.fsnn_test_data,
        snakemake.output.csv,
        snakemake.output.plot_webp,
    )
