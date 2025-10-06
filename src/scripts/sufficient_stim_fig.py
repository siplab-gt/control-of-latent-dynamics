import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.plot_style as psty


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

    # Example plot for the top subfigure
    ax_top.plot([1, 2, 4, 8])
    ax_top.set(title=df["task"][0])

    ref_condition = cfg["reference_params"]

    vmin, vmax = 0, 1
    # TODO: use 0, 1 for real, but using data range for now to see more variation
    vmin, vmax = df.R2_ctrl_Xhat_Xref_fsnn.min(), df.R2_ctrl_Xhat_Xref_fsnn.max()
    for ep, ax in zip(expl_params, axs_grid.flatten()):
        conds2keep = ref_condition.copy()
        del conds2keep[ep]
        query_str = " and ".join([f"{k} == {v}" for k, v in conds2keep.items()])
        subset = df.query(query_str)
        # average over seeds
        subset = (
            subset.groupby([ep, "k"])[["R2_ctrl_Xhat_Xref_fsnn"]].mean().reset_index()
        )
        subset = subset.pivot(index="k", columns=ep, values="R2_ctrl_Xhat_Xref_fsnn")
        sns.heatmap(subset, ax=ax, vmin=0, vmax=1, cmap=psty.cmap, cbar=False)

    for ax in axs_grid[:, 1:].flatten():
        ax.set(ylabel="")

    fig.colorbar(
        axs_grid[0, 0].collections[0],
        ax=axs_grid,
        label=r"$R^2\left(X_\text{ref} \sim X_\text{hat}\right)$",
    )
    for f in outfiles:
        print(f"Saving figure to {f}")
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.config, snakemake.input[0], snakemake.output)
