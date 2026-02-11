import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import CubicSpline

import src.plot_style as psty
from src.helpers import setup_debugger


def cubic_spline_interp(x, y):
    i_sort = np.argsort(x)
    assert np.all(np.diff(x[i_sort]) > 0), "x values must be unique for interpolation"
    cs = CubicSpline(x[i_sort], y[i_sort])
    return cs


def interp_argmax(x, y):
    cs = cubic_spline_interp(x, y)
    deriv = cs.derivative()
    extrema_candidates = deriv.roots()
    if len(extrema_candidates) == 0:
        return x[np.argmax(y)]  # No extrema found, return argmax of original data
    y_values = cs(extrema_candidates)
    x_max = extrema_candidates[y_values.argmax()]

    return x_max


def interp_argwhere(x, y, target_y):
    shifted_cs = cubic_spline_interp(x, y - target_y)
    roots = shifted_cs.roots()
    if len(roots) == 0:
        return np.nan  # No solution found
    return sorted(roots)[0]  # Return the smallest root


def compute_optimal_alpha_stim(subset_df, ep, metric, **kwargs):
    """
    Compute optimal alpha_stim for optimal_stim_rec analysis.

    Selects the alpha_stim values corresponding to maximum control quality (argmax),
    then computes alpha_stim = k / (m + k) for those selected rows.
    """
    total_channel_budget = subset_df.m + subset_df.k
    assert np.all(np.diff(total_channel_budget) == 0), (
        "Expected constant total channel budget across rows"
    )
    total_channel_budget = total_channel_budget.iloc[0]

    # Get alpha_stim values corresponding to max Q (argmax)
    def get_optimal_alpha_stim(group):
        x = group["k"].values
        y = group[metric].values
        k_opt = interp_argmax(x, y)
        return pd.Series({"optimal_alpha_stim": k_opt / total_channel_budget})

    return (
        subset_df.groupby([ep, "ctrl_latency", "seed"])
        .apply(get_optimal_alpha_stim, include_groups=False)
        .reset_index()
    )


def compute_k_suff(subset_df, ep, metric, ref_data, cfg, **kwargs):
    """
    Compute k_suff: k value where control quality is 90% of no-latency LQMPC max.

    For sufficient_stim analysis.
    """

    # find first k where Q >= Q_suff
    def find_k_suff(group):
        k_suff = interp_argwhere(
            group["k"].values,
            group[metric].values,
            target_y=0.9 * group[metric].values.max(),
        )

        return pd.Series({"k_suff": k_suff})

    return (
        subset_df.groupby([ep, "ctrl_latency", "seed"])
        .apply(find_k_suff, include_groups=False)
        .reset_index()
    )


def main(
    cfg,
    results_csv,
    outfiles,
    setting,
):
    expl_params = cfg["star_sweep_params"]
    if setting == "sufficient_stim":
        metric = "Q_track_Xref_X_fsnn"
        colorbar_label = r"$Q_\text{track}\left(X_\text{ref}, X\right)$"
        key_metric_name = "k_suff"
        key_metric_func = compute_k_suff
        main_fig_plot_type = "lineplots"
        plots_per_row = 3
    else:  # optimal_stim_rec
        assert setting == "optimal_stim_rec", f"Unknown setting: {setting}"
        metric = "Q_track_Zref_Z"
        colorbar_label = r"$Q_\text{track}\left(Z_\text{ref}, Z\right)$"
        key_metric_name = "optimal_alpha_stim"
        key_metric_func = compute_optimal_alpha_stim
        main_fig_plot_type = "lineplots"
        expl_params |= cfg["star_sweep_params_measurement_only"]
        plots_per_row = 4

    psty.style_plots()
    df = pd.read_csv(results_csv)
    df["alpha_stim"] = df.k / (df.m + df.k)

    # Create a combined key for hue
    df["ctrl_latency"] = df["ctrl_type"] + df["latency"].astype(str)
    df["ctrl_latency"] = df["ctrl_latency"].map(
        {"OLFalse": "OL", "LQMPCFalse": "MPC", "LQMPCTrue": "MPC with latency"}
    )
    rows = int(np.ceil(len(expl_params) / plots_per_row))

    def grid_subplots(extra_row=False):
        n_rows = rows + 1 if extra_row else rows
        fig, axs = plt.subplots(
            n_rows,
            plots_per_row,
            figsize=(psty.double_col_width, 2 * n_rows),
            layout="compressed",
            sharey=True,
        )
        btm_axs = axs[1:, :] if extra_row else axs
        for i in range(len(expl_params), len(btm_axs.flatten())):
            btm_axs.flatten()[i].axis("off")
        return fig, axs

    main_fig, main_axs = grid_subplots(extra_row=True)
    # First row: single subfigure spanning all columns
    ax_top = plt.subplot2grid((1 + rows, plots_per_row), (0, 0), colspan=plots_per_row)
    # Remove axes in first row (since we use ax_top)
    for ax in main_axs[0]:
        ax.remove()
    axs_grid = main_axs[1:]

    ref_condition = cfg["reference_params"]

    # top plot
    ref_cond_except_latency = ref_condition.copy()
    del ref_cond_except_latency["latency"]
    ref_data = df.query(
        " and ".join(
            [
                f"({k} == {v!r} or {k}.isnull())"
                for k, v in ref_cond_except_latency.items()
            ]
        )
    ).copy()

    def _interp(group):
        x = group["k"].values
        y = group[metric].values
        k_interp = np.arange(x.min(), x.max() + 1)
        cs = cubic_spline_interp(x, y)
        y_interp = cs(k_interp)
        return pd.DataFrame({"k": k_interp, metric: y_interp})

    interp_ref_data = ref_data.groupby(["ctrl_latency", "seed"]).apply(
        _interp, include_groups=False
    )

    # dots to show actual data
    sns.scatterplot(
        data=ref_data,
        x="k",
        y=metric,
        hue="ctrl_latency",
        ax=ax_top,
        marker=".",
        lw=0.4,
        legend=False,
        palette=psty.ctrl_palette,
    )
    # thin lines show interpolated per-seed data
    sns.lineplot(
        data=interp_ref_data,
        x="k",
        y=metric,
        hue="ctrl_latency",
        ax=ax_top,
        marker=None,
        errorbar=None,
        units="seed",
        estimator=None,
        lw=0.4,
        legend=False,
        palette=psty.ctrl_palette,
    )
    # thick lines show average of interpolated data
    sns.lineplot(
        data=interp_ref_data,
        x="k",
        y=metric,
        hue="ctrl_latency",
        ax=ax_top,
        errorbar=None,
        lw=2,
        palette=psty.ctrl_palette,
    )

    def plot_lineplots(lp_axs):
        """Plot lineplots showing key metric vs exploration parameter."""
        for ep, ax in zip(expl_params, lp_axs.flatten()):
            conds2keep = ref_condition.copy()
            del conds2keep[ep], conds2keep["latency"]
            query_str = " and ".join(
                [f"({k} == {v!r} or {k}.isnull())" for k, v in conds2keep.items()]
            )
            query_str += ' and (ctrl_latency == "MPC")'
            subset = df.query(query_str)
            assert len(subset[ep].unique()) > 1, (
                f"No variation in {ep} in subset. Values: {subset[ep].unique()}"
            )
            # Apply key metric computation function
            subset = key_metric_func(subset, ep, metric, ref_data=ref_data, cfg=cfg)
            sns.lineplot(
                data=subset,
                x=ep,
                y=key_metric_name,
                hue="ctrl_latency",
                ax=ax,
                legend=False,
                palette=psty.ctrl_palette,
                **psty.sns_errorbar_kwargs,
                marker="D",
            )
            # ax.set(ylim=(0, 1))
        for ax in lp_axs[:, 1:].flatten():
            ax.set(ylabel="")

    def plot_heatmaps(hm_fig, hm_axs, mean_or_std):
        vmin, vmax = df[metric].min(), df[metric].max()
        vmin = max(vmin, -1)
        vmax = min(vmax, 1)
        heatmap_kwargs = dict(
            mean=dict(
                vmin=vmin,
                vmax=vmax,
                center=0,
                cmap=psty.div_cmap,
            ),
            std=dict(cmap=psty.cmap),
        )[mean_or_std]
        for ep, ax in zip(expl_params, hm_axs.flatten()):
            conds2keep = ref_condition.copy()
            del conds2keep[ep]
            query_str = " and ".join(
                [f"({k} == {v!r} or {k}.isnull())" for k, v in conds2keep.items()]
            )
            subset = df.query(query_str)
            # average over seeds
            subset = subset.groupby([ep, "k"])[[metric]]
            subset = getattr(subset, mean_or_std)().reset_index()
            subset = subset.pivot(index="k", columns=ep, values=metric)
            sns.heatmap(
                subset,
                ax=ax,
                cbar=False,
                **heatmap_kwargs,
            )
            ax.invert_yaxis()
            ax.set_title(ep)

            # Add text annotations for extreme values not represented by colormap
            for i, k_val in enumerate(subset.index):
                for j, ep_val in enumerate(subset.columns):
                    val = subset.iloc[i, j]
                    if val < vmin:
                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=8,
                        )

        for ax in hm_axs[:, 1:].flatten():
            ax.set(ylabel="")

        hm_fig.colorbar(
            hm_axs[0, 0].collections[0],
            ax=hm_axs,
            label=mean_or_std + " " + colorbar_label,
        )

    # main fig: choose between lineplots and heatmaps for the grid
    if main_fig_plot_type == "lineplots":
        plot_lineplots(axs_grid)
    else:  # heatmaps
        plot_heatmaps(main_fig, axs_grid, "mean")

    fig_lp, axs_lp = grid_subplots()
    plot_lineplots(axs_lp)

    fig_hm_mean, axs_hm_mean = grid_subplots()
    plot_heatmaps(fig_hm_mean, axs_hm_mean, "mean")

    fig_hm_std, axs_hm_std = grid_subplots()
    plot_heatmaps(fig_hm_std, axs_hm_std, "std")

    for fig, files in [
        (main_fig, outfiles.main),
        (fig_lp, outfiles.lineplots),
        (fig_hm_mean, outfiles.heatmaps_mean),
        (fig_hm_std, outfiles.heatmaps_std),
    ]:
        for f in files:
            fig.savefig(f)
            print(f"Saving {f}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config.get("debug", False))

    main(
        snakemake.config,
        snakemake.input[0],
        snakemake.output,
        setting=snakemake.wildcards.setting,
    )


if __name__ == "__main__" and "snakemake" not in locals():
    import argparse
    import pathlib

    import snakemake

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-csv",
        required=True,
        help="CSV file with results to plot",
    )
    args = parser.parse_args()

    setting_task = pathlib.Path(args.results_csv).stem
    setting = "_".join(setting_task.split("_")[:-1])
    print(f"{setting_task=}")

    exts = [".webp", ".pdf"]

    add_exts = lambda base: [f"{base}{ext}" for ext in exts]

    outfiles = argparse.Namespace(
        main=add_exts(f"results/figs/{setting_task}"),
        lineplots=add_exts(f"results/figs/{setting_task}_lineplots"),
        heatmaps_mean=add_exts(f"results/figs/{setting_task}_heatmaps_mean"),
        heatmaps_std=add_exts(f"results/figs/{setting_task}_heatmaps_std"),
    )
    print(f"{outfiles=}")

    cfg = snakemake.common.configfile._load_configfile("./config/config.yaml")

    main(cfg, args.results_csv, outfiles, setting=setting)
