import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.plot_style as psty
from src.helpers import setup_debugger


def compute_optimal_p_stim(subset_df, ep, metric, **kwargs):
    """
    Compute optimal p_stim for optimal_stim_rec analysis.

    Selects the p_stim values corresponding to maximum control quality (argmax),
    then computes p_stim = k / (m + k) for those selected rows.
    """
    # Get p_stim values corresponding to max Q (argmax)
    result = subset_df.loc[
        subset_df.groupby([ep, "ctrl_latency", "seed"])[metric].idxmax()
    ].reset_index(drop=True)
    result["optimal_p_stim"] = result["p_stim"]
    return result


def compute_k_suff(subset_df, ep, metric, ref_data, cfg, **kwargs):
    """
    Compute k_suff: k value where control quality is 90% of no-latency LQMPC max.

    For sufficient_stim analysis.
    """
    # get Q_suff from the Qs per seed
    ref_data = (
        ref_data.groupby(["k", "ctrl_latency", "seed"])[[metric]].mean().reset_index()
    )
    # will probably be no-latency LQMPC, max k
    # Q_suff_per_seed = (
    #     cfg["q1"]["sufficient_ctrl"] * ref_data.groupby("seed")[metric].max()
    # )

    # find first k where Q >= Q_suff
    def find_k_suff(group):
        Q_suff = group[metric].max() * cfg["q1"]["sufficient_ctrl"]

        suff_rows = group[group[metric] >= Q_suff]
        if suff_rows.empty:
            return pd.Series({"k_suff": float("nan")})

        first_sufficient_idx = suff_rows.index[0]
        if first_sufficient_idx == group.index[0]:
            # First point is already sufficient
            return pd.Series({"k_suff": suff_rows.k.min()})

        # Get last insufficient and first sufficient points
        last_insufficient = group.loc[:first_sufficient_idx].iloc[-2]
        first_sufficient = group.loc[first_sufficient_idx]

        # Linear interpolation
        k1, q1 = last_insufficient["k"], last_insufficient[metric]
        k2, q2 = first_sufficient["k"], first_sufficient[metric]
        k_suff = k1 + (Q_suff - q1) * (k2 - k1) / (q2 - q1)

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
        key_metric_name = "optimal_p_stim"
        key_metric_func = compute_optimal_p_stim
        main_fig_plot_type = "lineplots"
        expl_params |= cfg["star_sweep_params_measurement_only"]
        plots_per_row = 4

    psty.style_plots()
    df = pd.read_csv(results_csv)
    df["p_stim"] = df.k / (df.m + df.k)

    # Create a combined key for hue
    df["ctrl_latency"] = df["ctrl_type"] + df["latency"].astype(str)
    df["ctrl_latency"] = df["ctrl_latency"].map(
        {"OLFalse": "OL", "LQMPCFalse": "LQMPC no latency", "LQMPCTrue": "LQMPC"}
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
        for i in range(len(expl_params), len(axs.flatten())):
            axs.flatten()[i].axis("off")
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

    sns.lineplot(
        data=ref_data,
        x="k",
        y=metric,
        hue="ctrl_latency",
        ax=ax_top,
        marker=".",
        errorbar=None,
        units="seed",
        estimator=None,
        lw=0.4,
        legend=False,
        palette=psty.ctrl_palette,
    )
    sns.lineplot(
        data=ref_data,
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
            query_str += ' and (ctrl_latency != "LQMPC no latency")'
            subset = df.query(query_str)
            assert len(subset[ep].unique()) > 1, f"No variation in {ep} in subset"
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
        vmin, vmax = 0, 1
        # TODO: use 0, 1 for real, but using data range for now to see more variation
        vmin, vmax = df[metric].min(), df[metric].max()
        # TODO: limit vmin, vmax to reasonable range
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
