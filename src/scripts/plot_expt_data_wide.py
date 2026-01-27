import matplotlib.pyplot as plt
import numpy as np

from src.helpers import setup_debugger
from src.plot_style import cmap, style_plots
from src.scripts.sys_id_expts import ExptData


def plot_expt_data_wide(expt_data: ExptData, outfile, style="default"):
    assert style in ["default", "heatmap", "exploded"]
    varnames = ["X", "Z", "spikes", "lfp", "Uopto", "Utask"]
    nrows = len(varnames)
    figwidth = 1 * expt_data.n_trials
    if style == "exploded":
        ht_per_series = 0.3
        num_series_per_row = []
        for varname in varnames:
            data = getattr(expt_data, varname)
            if data is not None and data.size > 0:
                num_series_per_row.append(data.shape[1])
            else:
                num_series_per_row.append(1)
        figheight = ht_per_series * sum(num_series_per_row)
        kwargs = {"height_ratios": num_series_per_row}
    else:
        figheight = 2 * nrows
        kwargs = {}
    fig, axs = plt.subplots(
        nrows,
        1,
        figsize=(figwidth, figheight),
        sharex=True,
        layout="compressed",
        **kwargs,
    )
    for ax, varname, t in zip(
        axs,
        varnames,
        [
            expt_data.t_ms,
            expt_data.t_ms,
            expt_data.t_samp_ms,
            expt_data.t_samp_ms,
            expt_data.t_opto_ms,
            expt_data.t_ms,
        ],
    ):
        data = getattr(expt_data, varname)
        if data is None or data.size == 0:
            print(f"skipping {varname} since empty")
            continue
        print(f"plotting {varname}")
        offsets = (
            np.arange(data.shape[1]) * np.percentile(np.ptp(data), 60)
            if style == "exploded"
            else 0
        )
        if style in ["default", "exploded"]:
            ax.plot(t, data + offsets)
            ax.set(ylabel=varname)
        elif style == "heatmap":
            im = ax.imshow(
                data.T,
                aspect="auto",
                extent=[t[0], t[-1], -0.5, data.shape[1] - 0.5],
                origin="lower",
                cmap="viridis",
                interpolation="nearest",
            )
            ax.set(ylabel=varname)
            # ensure no sub-integer yticks
            yticks = ax.get_yticks()
            ax.set_yticks(
                yticks[(yticks % 1 == 0) & (yticks >= 0) & (yticks < data.shape[1])]
            )
            fig.colorbar(im, ax=ax)
    axs[-1].set(xlabel="time (ms)")
    for i, (i_start, _) in enumerate(expt_data.i_trial_boundaries):
        t_start = expt_data.t_ms[i_start]
        axs[0].annotate(
            f"trial {i}",
            xy=(t_start, axs[0].get_ylim()[1]),
            xycoords="data",
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            annotation_clip=False,
        )

    if outfile:
        plt.savefig(outfile)
    return fig, axs


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    style_plots()
    expt_data = ExptData.load(snakemake.input[0])
    style = snakemake.wildcards.get("style", "default")
    plot_expt_data_wide(expt_data, snakemake.output[0], style=style)
