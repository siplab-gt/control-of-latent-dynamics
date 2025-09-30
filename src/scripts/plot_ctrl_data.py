import matplotlib.pyplot as plt
import numpy as np

from src.plot_style import style_plots
from src.scripts.ctrl_expts import CtrlData


def plot_all(ctrl_data: CtrlData, outfile, explode=False):
    vars = ["X_fsnn", "Xhat_lds", "Z", "spikes", "Uopto", "Utask"]
    dashed_vars = ["Xref_fsnn", "Xref_lds", "Zref", "Yref", None, None]
    light_vars = ["Xhat_fsnn", None, "Zhat", "Yhat", None, None]
    nrows = len(vars)
    figwidth = 1 * ctrl_data.n_trials
    if not explode:
        figheight = 2 * nrows
        kwargs = {}
    elif explode:
        ht_per_series = 0.3
        num_series_per_row = [getattr(ctrl_data, v).shape[1] for v in vars]
        figheight = ht_per_series * sum(num_series_per_row)
        kwargs = {"height_ratios": num_series_per_row}
    fig, axs = plt.subplots(
        nrows,
        1,
        figsize=(figwidth, figheight),
        sharex=True,
        layout="compressed",
        **kwargs,
    )
    for ax, varname, t, dashed_varname, light_varname in zip(
        axs,
        vars,
        [
            ctrl_data.t_ms,
            ctrl_data.t_ms,
            ctrl_data.t_ms,
            ctrl_data.t_samp_ms,
            ctrl_data.t_opto_ms,
            ctrl_data.t_ms,
        ],
        dashed_vars,
        light_vars,
    ):
        print(f"plotting {varname}")
        data = getattr(ctrl_data, varname)
        offsets = (
            np.arange(data.shape[1]) * np.percentile(np.ptp(data), 60) if explode else 0
        )

        ax.plot(t, data + offsets)
        ax.set_prop_cycle(None)  # reset color cycler
        if dashed_varname is not None:
            print(f" with {dashed_varname}")
            ax.plot(t, getattr(ctrl_data, dashed_varname) + offsets, ls="--")
            ax.set_prop_cycle(None)  # reset color cycler
        if light_varname is not None:
            print(f" and {light_varname}")
            ax.plot(t, getattr(ctrl_data, light_varname) + offsets, alpha=0.5)
        ax.set(
            title=f"{varname}, {dashed_varname} dashed, {light_varname} light",
            ylabel=varname,
        )

    axs[-1].set(xlabel="time (ms)")
    for i, (i_start, _) in enumerate(ctrl_data.i_trial_boundaries):
        t_start = ctrl_data.t_ms[i_start]
        axs[0].annotate(
            f"trial {i}",
            xy=(t_start, axs[0].get_ylim()[1]),
            xycoords="data",
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="top",
            annotation_clip=False,
        )
        for ax in axs:
            ax.axvline(t_start, color="xkcd:light gray", lw=0.5)

    plt.savefig(outfile)
    return fig, axs


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    style_plots()
    expt_data = CtrlData.load(snakemake.input[0])
    plot_all(expt_data, snakemake.output[0])
    plot_all(expt_data, snakemake.output[1], explode=True)
