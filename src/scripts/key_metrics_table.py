import os

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

from src.helpers import preprocess_params, setup_debugger


def interpolator(x, y):
    i_sort = x.argsort()
    assert (x[i_sort][1:] - x[i_sort][:-1] > 0).all(), (
        "x values must be unique for interpolation"
    )
    return make_interp_spline(x[i_sort], y[i_sort], k=1)


def interp_argwhere(x, y, target_y):
    """returns first x where interpolation equals target y within
    range of input x"""
    shifted_interp = interpolator(x, y - target_y)
    new_x = np.linspace(x.min(), x.max(), 10000)
    new_y = shifted_interp(new_x)
    ans = new_x[np.argmin(np.abs(new_y))]
    assert x.min() <= ans <= x.max(), (
        "Interpolated k must be within the range of k values"
    )
    return ans


def compute_optimal_alpha_stim(subset_df, ep, metric, **kwargs):
    total_channel_budget = subset_df.m + subset_df.k
    assert (total_channel_budget.diff().fillna(0) == 0).all(), (
        "Expected constant total channel budget across rows"
    )
    total_channel_budget = total_channel_budget.iloc[0]

    def get_optimal_alpha_stim(group):
        x = group["k"].to_numpy()
        y = group[metric].to_numpy()
        k_opt = x[y.argmax()]
        assert x.min() <= k_opt <= x.max(), (
            "Optimal k must be within the range of k values"
        )
        return pd.Series({"optimal_alpha_stim": k_opt / total_channel_budget})

    out = (
        subset_df.groupby([ep, "ctrl_latency", "seed"])
        .apply(get_optimal_alpha_stim, include_groups=False)
        .reset_index()
    )
    assert out["optimal_alpha_stim"].between(0, 1).all(), (
        "Optimal alpha stim must be between 0 and 1"
    )
    return out


def compute_k_suff(subset_df, ep, metric, cfg=None):
    target_frac = cfg["q1"]["sufficient_ctrl"]

    def find_k_suff(group):
        k_suff = interp_argwhere(
            group["k"].to_numpy(),
            group[metric].to_numpy(),
            target_y=target_frac * group[metric].max(),
        )
        return pd.Series({"k_0.9": k_suff})

    return (
        subset_df.groupby([ep, "ctrl_latency", "seed"])
        .apply(find_k_suff, include_groups=False)
        .reset_index()
    )


def build_key_metrics_table(cfg, base_params, results_csv, wildcards):
    expl_params = (
        cfg["star_sweep_params"].copy() | cfg[f"star_sweep_params_{wildcards.setting}"]
    )

    ctrl_metric = base_params.ctrl_metric
    key_metric = base_params.key_metric
    if key_metric == "k_0.9":
        key_metric_func = compute_k_suff
    else:
        assert key_metric == "optimal_alpha_stim", f"Unknown key metric: {key_metric}"
        key_metric_func = compute_optimal_alpha_stim

    df = pd.read_csv(results_csv)
    df["alpha_stim"] = df.k / (df.m + df.k)

    df["ctrl_latency"] = df["ctrl_type"] + df["latency"].astype(str)
    df["ctrl_latency"] = df["ctrl_latency"].map(
        {"OLFalse": "OL", "LQMPCFalse": "MPC", "LQMPCTrue": "MPC with latency"}
    )

    ref_condition = cfg["reference_params"]
    ref_condition |= cfg[f"reference_params_{wildcards.setting}"]
    rows = []

    for ep in expl_params:
        conds2keep = ref_condition.copy()
        del conds2keep[ep], conds2keep["latency"]
        query_str = " and ".join(
            [f"({k} == {v!r} or {k}.isnull())" for k, v in conds2keep.items()]
        )
        query_str += ' and (ctrl_latency == "MPC")'
        subset = df.query(query_str)
        if subset.empty:
            continue

        metric_df = key_metric_func(subset, ep, ctrl_metric, cfg=cfg)

        metric_df["ep"] = ep
        metric_df["ep_value"] = metric_df[ep]
        metric_df["setting"] = wildcards.setting
        metric_df["task"] = wildcards.task
        rows.append(
            metric_df[
                [
                    "setting",
                    "task",
                    "ep",
                    "ep_value",
                    "seed",
                    "ctrl_latency",
                    key_metric,
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "setting",
                "task",
                "ep",
                "ep_value",
                "seed",
                "ctrl_latency",
                key_metric,
            ]
        )

    return pd.concat(rows, ignore_index=True)


def main(cfg, base_params, results_csv, outfile, wildcards):
    key_metrics = build_key_metrics_table(cfg, base_params, results_csv, wildcards)
    key_metrics.to_csv(outfile, index=False)
    print(f"Wrote key metrics to {outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config.get("debug", False))

    main(
        snakemake.config,
        preprocess_params(snakemake.params),
        snakemake.input[0],
        snakemake.output[0],
        wildcards=snakemake.wildcards,
    )


if __name__ == "__main__" and "snakemake" not in locals():
    import argparse

    import snakemake

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = snakemake.common.configfile._load_configfile("./config/config.yaml")
    setting, task = os.path.basename(args.results_csv).split("_")[0:2]

    main(
        cfg,
        preprocess_params(cfg[f"analysis_{setting}"]),
        args.results_csv,
        args.out,
        wildcards=dict(setting=setting, task=task),
    )
