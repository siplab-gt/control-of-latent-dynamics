import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.plot_style as psty


def main(cfg, results_csv, outfiles):
    psty.style_plots()
    df = pd.read_csv(results_csv)
    df["p_stim"] = df.k / (df.m + df.k)
    # Create a combined key for hue
    df["ctrl_latency"] = df["ctrl_type"] + df["latency"].map(
        {True: ", w/ latency", False: ""}
    )
    expl_params = cfg["star_sweep_params"]
    plots_per_row = 3
    rows = (len(expl_params) + plots_per_row - 1) // plots_per_row

    fig, axs = plt.subplots(
        1 + rows,
        plots_per_row,
        figsize=(psty.double_col_width, 2 * (1 + rows)),
        layout="compressed",
        sharey="row",
    )
    # First row: single subfigure spanning all columns
    ax_top = plt.subplot2grid((1 + rows, plots_per_row), (0, 0), colspan=plots_per_row)
    # Remove axes in first row (since we use ax_top)
    for ax in axs[0]:
        ax.remove()
    axs_grid = axs[1:]

    ref_condition = cfg["reference_params"]

    # top plot
    metric = "Q_track_Zref_Z"
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
        # size="ctrl_latency",
        ax=ax_top,
        marker=".",
        # errorbar=("pi", 95),
        errorbar=None,
        units="seed",
        estimator=None,
        lw=0.4,
        legend=False,
    )
    sns.lineplot(
        data=ref_data,
        x="k",
        y=metric,
        hue="ctrl_latency",
        # size="ctrl_latency",
        ax=ax_top,
        # marker="o",
        # errorbar=("pi", 95),
        errorbar=None,
        # units="seed",
        # estimator=None,
        lw=2,
    )

    def plot_lineplots(lp_axs):
        ylim = (0, 1)
        for ep, ax in zip(expl_params, lp_axs.flatten()):
            conds2keep = ref_condition.copy()
            del conds2keep[ep], conds2keep["latency"]
            query_str = " and ".join(
                [f"({k} == {v!r} or {k}.isnull())" for k, v in conds2keep.items()]
            )
            subset = df.query(query_str)
            assert len(subset[ep].unique()) > 1, f"No variation in {ep} in subset"
            # get p_stim values corresponding to max Q (argmax)
            # grouped = subset.groupby([ep, "ctrl_latency", "seed"])[metric]
            # print(f"{ep} has groups {list(grouped.groups.keys())}")
            subset = subset.loc[
                subset.groupby([ep, "ctrl_latency", "seed"])[metric].idxmax()
            ].reset_index(drop=True)
            # print(subset)
            sns.lineplot(
                data=subset,
                x=ep,
                y="p_stim",
                hue="ctrl_latency",
                ax=ax,
                legend=False,
                **psty.sns_errorbar_kwargs,
            )
            ax.set(ylim=ylim)
        for ax in lp_axs[:, 1:].flatten():
            ax.set(ylabel="")

    def plot_heatmaps(hm_fig, hm_axs, mean_or_std):
        vmin, vmax = 0, 1
        # TODO: use 0, 1 for real, but using data range for now to see more variation
        vmin, vmax = df[metric].min(), df[metric].max()
        # TODO: limit vmin, vmax to reasonable range
        # vmin = max(vmin, -1)
        # vmax = min(vmax, 1)
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

        for ax in hm_axs[:, 1:].flatten():
            ax.set(ylabel="")

        hm_fig.colorbar(
            hm_axs[0, 0].collections[0],
            ax=hm_axs,
            label=mean_or_std + r" $Q_\text{track}\left(Z_\text{ref}, Z\right)$",
        )

    # main fig
    plot_lineplots(axs_grid)
    for f in outfiles:
        print(f"Saving figure to {f}")
        fig.savefig(f)

    fig_hm_mean, axs_hm_mean = plt.subplots(
        rows,
        plots_per_row,
        figsize=(psty.double_col_width, 2 * (rows)),
        layout="compressed",
        sharey="row",
    )
    plot_heatmaps(fig_hm_mean, axs_hm_mean, "mean")
    for f in outfiles.heatmaps_mean:
        fig_hm_mean.savefig(f)

    fig_hm_std, axs_hm_std = plt.subplots(
        rows,
        plots_per_row,
        figsize=(psty.double_col_width, 2 * (rows)),
        layout="compressed",
        sharey="row",
    )
    plot_heatmaps(fig_hm_std, axs_hm_std, "std")
    for f in outfiles.heatmaps_std:
        fig_hm_std.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.config, snakemake.input[0], snakemake.output)
