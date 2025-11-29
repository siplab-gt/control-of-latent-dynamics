import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.plot_style as psty
from src.helpers import setup_debugger


def main(cfg, results_csv, outfiles):
    psty.style_plots()
    df = pd.read_csv(results_csv)
    expl_params = cfg["star_sweep_params"]
    plots_per_row = 4
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

    # Example plot for the top subfigure
    metric = "Q_track_Xref_X_fsnn"
    # top plot
    ref_cond_except_latency = ref_condition.copy()
    del ref_cond_except_latency["latency"]
    ref_data = df.query(
        # need k.isnull() for variables that don't apply (signals in Q1)
        " and ".join(
            [
                f"({k} == {v!r} or {k}.isnull())"
                for k, v in ref_cond_except_latency.items()
            ]
        )
    ).copy()

    # Create a combined key for hue
    ref_data["ctrl_latency"] = ref_data["ctrl_type"] + ref_data["latency"].map(
        {True: ", w/ latency", False: ""}
    )
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

    # vmin, vmax = 0, 1
    # TODO: use 0, 1 for real, but using data range for now to see more variation
    vmin, vmax = df[metric].min(), df[metric].max()
    # TODO
    vmin = max(vmin, -1)
    vmax = min(vmax, 1)
    for ep, ax in zip(expl_params, axs_grid.flatten()):
        conds2keep = ref_condition.copy()
        del conds2keep[ep]
        query_str = " and ".join(
            [f"({k} == {v!r} or {k}.isnull())" for k, v in conds2keep.items()]
        )
        subset = df.query(query_str)
        # average over seeds
        subset = subset.groupby([ep, "k"])[[metric]].mean().reset_index()
        subset = subset.pivot(index="k", columns=ep, values=metric)
        sns.heatmap(
            subset,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            center=0,
            cmap=psty.div_cmap,
            cbar=False,
        )

    for ax in axs_grid[:, 1:].flatten():
        ax.set(ylabel="")

    fig.colorbar(
        axs_grid[0, 0].collections[0],
        ax=axs_grid,
        label=r"$Q_\text{track}\left(X_\text{ref}, X\right)$",
    )
    for f in outfiles:
        print(f"Saving figure to {f}")
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config["debug"])

    main(snakemake.config, snakemake.input[0], snakemake.output)
