import pandas as pd
from scipy.interpolate import CubicSpline

from src.helpers import setup_debugger


def cubic_spline_interp(x, y):
    i_sort = x.argsort()
    assert (x[i_sort][1:] - x[i_sort][:-1] > 0).all(), (
        "x values must be unique for interpolation"
    )
    return CubicSpline(x[i_sort], y[i_sort])


def interp_argmax(x, y):
    cs = cubic_spline_interp(x, y)
    deriv = cs.derivative()
    extrema_candidates = deriv.roots()
    if len(extrema_candidates) == 0:
        return x[y.argmax()]  # No extrema found, return argmax of original data
    y_values = cs(extrema_candidates)
    return extrema_candidates[y_values.argmax()]


def interp_argwhere(x, y, target_y):
    """returns first x where interpolation equals target y within
    range of input x"""
    shifted_cs = cubic_spline_interp(x, y - target_y)
    roots = shifted_cs.roots()
    roots = roots[(roots >= x.min()) & (roots <= x.max())]  # Filter roots within bounds
    if len(roots) == 0:
        return float("nan")  # No solution found
    ans = sorted(roots)[0]
    return ans  # Return the smallest root


def compute_optimal_alpha_stim(subset_df, ep, metric):
    total_channel_budget = subset_df.m + subset_df.k
    assert (total_channel_budget.diff().fillna(0) == 0).all(), (
        "Expected constant total channel budget across rows"
    )
    total_channel_budget = total_channel_budget.iloc[0]

    def get_optimal_alpha_stim(group):
        x = group["k"].to_numpy()
        y = group[metric].to_numpy()
        k_opt = interp_argmax(x, y)
        return pd.Series({"optimal_alpha_stim": k_opt / total_channel_budget})

    return (
        subset_df.groupby([ep, "ctrl_latency", "seed"])
        .apply(get_optimal_alpha_stim, include_groups=False)
        .reset_index()
    )


def compute_k_suff(subset_df, ep, metric, cfg):
    target_frac = cfg["q1"]["sufficient_ctrl"]

    def find_k_suff(group):
        k_suff = interp_argwhere(
            group["k"].to_numpy(),
            group[metric].to_numpy(),
            target_y=target_frac * group[metric].max(),
        )
        return pd.Series({"k_suff": k_suff})

    return (
        subset_df.groupby([ep, "ctrl_latency", "seed"])
        .apply(find_k_suff, include_groups=False)
        .reset_index()
    )


def build_key_metrics_table(cfg, results_csv, setting, task):
    expl_params = cfg["star_sweep_params"].copy()

    if setting == "sufficient_stim":
        metric = "Q_track_Xref_X_fsnn"
        key_metric_name = "k_suff"
        key_metric_func = compute_k_suff
    else:
        assert setting == "optimal_stim_rec", f"Unknown setting: {setting}"
        metric = "Q_track_Zref_Z"
        key_metric_name = "optimal_alpha_stim"
        key_metric_func = compute_optimal_alpha_stim
        expl_params |= cfg["star_sweep_params_measurement_only"]

    df = pd.read_csv(results_csv)
    df["alpha_stim"] = df.k / (df.m + df.k)

    df["ctrl_latency"] = df["ctrl_type"] + df["latency"].astype(str)
    df["ctrl_latency"] = df["ctrl_latency"].map(
        {"OLFalse": "OL", "LQMPCFalse": "MPC", "LQMPCTrue": "MPC with latency"}
    )

    ref_condition = cfg["reference_params"]
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

        if key_metric_name == "k_suff":
            metric_df = key_metric_func(subset, ep, metric, cfg=cfg)
        else:
            metric_df = key_metric_func(subset, ep, metric)

        metric_df = metric_df.rename(columns={key_metric_name: "key_metric_value"})
        metric_df["key_metric_name"] = key_metric_name
        metric_df["ep"] = ep
        metric_df["ep_value"] = metric_df[ep]
        metric_df["setting"] = setting
        metric_df["task"] = task
        rows.append(
            metric_df[
                [
                    "setting",
                    "task",
                    "ep",
                    "ep_value",
                    "seed",
                    "ctrl_latency",
                    "key_metric_name",
                    "key_metric_value",
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
                "key_metric_name",
                "key_metric_value",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def main(cfg, results_csv, outfile, setting, task):
    key_metrics = build_key_metrics_table(cfg, results_csv, setting, task)
    key_metrics.to_csv(outfile, index=False)
    print(f"Wrote key metrics to {outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config.get("debug", False))

    main(
        snakemake.config,
        snakemake.input[0],
        snakemake.output[0],
        setting=snakemake.wildcards.setting,
        task=snakemake.wildcards.task,
    )


if __name__ == "__main__" and "snakemake" not in locals():
    import argparse

    import snakemake

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = snakemake.common.configfile._load_configfile("./config/config.yaml")

    main(
        cfg,
        args.results_csv,
        args.out,
        setting=args.setting,
        task=args.task,
    )
