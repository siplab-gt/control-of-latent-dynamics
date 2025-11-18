import matplotlib.pyplot as plt

from src.helpers import setup_debugger
from src.plot_style import cmap, style_plots
from src.scripts.sys_id_expts import ExptData


def plot_expt_data_wide(expt_data: ExptData, outfile, style="default"):
    assert style in ["default", "heatmap"]
    varnames = ["X", "Z", "spikes", "lfp", "Uopto", "Utask"]
    nrows = len(varnames)
    figwidth = 1 * expt_data.n_trials
    figheight = 2 * nrows
    fig, axs = plt.subplots(nrows, 1, figsize=(figwidth, figheight), sharex=True)
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
        if style == "default":
            ax.plot(t, data)
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
