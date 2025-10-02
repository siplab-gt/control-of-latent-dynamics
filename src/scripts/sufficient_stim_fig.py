import matplotlib.pyplot as plt
import pandas as pd

import src.plot_style as psty


def main(cfg, results_csv, outfile):
    psty.style_plots()
    df = pd.read_csv(results_csv)
    expl_params = cfg["star_sweep_params"]
    plots_per_row = 4
    rows = (len(expl_params) + plots_per_row - 1) // plots_per_row

    fig, axs = plt.subplots(
        1 + rows,
        plots_per_row,
        figsize=(psty.double_col_width, 2 * (1 + rows)),
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

    # Example plots for the 4x3 grid
    for i in range(rows):
        for j in range(plots_per_row):
            axs_grid[i][j].plot([1, 2, 3], [i, j, i + j])
            axs_grid[i][j].set_title(f"Subplot {i * plots_per_row + j + 1}")

    fig.tight_layout()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 4, 8])
    ax.set(title=df["task"][0])
    fig.savefig(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.config, snakemake.input[0], snakemake.output[0])
