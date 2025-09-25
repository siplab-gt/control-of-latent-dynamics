import matplotlib.pyplot as plt

from src.plot_style import style_plots
from src.scripts.sys_id_expts import ExptData


def plot_all(expt_data: ExptData, outfile):
    vars = ["X", "Z", "spikes", "Uopto", "Utask"]
    nrows = len(vars)
    figwidth = 1 * expt_data.n_trials
    figheight = 2 * nrows
    fig, axs = plt.subplots(nrows, 1, figsize=(figwidth, figheight), sharex=True)
    for ax, varname, t in zip(
        axs,
        vars,
        [
            expt_data.t_ms,
            expt_data.t_ms,
            expt_data.t_samp_ms,
            expt_data.t_opto_ms,
            expt_data.t_ms,
        ],
    ):
        print(f"plotting {varname}")
        ax.plot(t, getattr(expt_data, varname))
        ax.set(ylabel=varname)
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

    plt.savefig(outfile)
    return fig, axs


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    style_plots()
    expt_data = ExptData.load(snakemake.input[0])
    plot_all(expt_data, snakemake.output[0])
