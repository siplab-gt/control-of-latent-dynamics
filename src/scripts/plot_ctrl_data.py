import matplotlib.pyplot as plt
import numpy as np

from src.helpers import setup_debugger
from src.plot_style import style_plots
from src.scripts.ctrl_expts import CtrlData


def plot_all(ctrl_data: CtrlData, outfile, setting="", explode=False):
    vars = ["X_fsnn", "Xhat_lds", "Z", "spikes", "Uopto", "Utask"]
    dashed_vars = ["Xref_fsnn", "Xref_lds", "Zref", "Yref", None, None]
    light_vars = ["Xhat_fsnn", None, "Zhat", "Yhat", None, None]

    to_ignore = dict(
        sufficient_stim=[], optimal_stim_rec=["Xref_fsnn", "Xref_lds", "Yref"]
    )[setting]

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
        if varname in to_ignore:
            print(f"skipping {varname} for setting {setting}")
            continue
        print(f"plotting {varname}")
        data = getattr(ctrl_data, varname)
        if data.size == 0:
            print(f"skipping {varname} since empty")
            continue
        offsets = (
            np.arange(data.shape[1]) * np.percentile(np.ptp(data), 60) if explode else 0
        )

        dashed_varname = dashed_varname if dashed_varname not in to_ignore else None
        light_varname = light_varname if light_varname not in to_ignore else None

        ax.plot(t, data + offsets)
        ax.set_prop_cycle(None)  # reset color cycler
        if dashed_varname is not None:
            print(f" with {dashed_varname}")
            ax.plot(t, getattr(ctrl_data, dashed_varname) + offsets, c="k", ls="--")
            ax.set_prop_cycle(None)  # reset color cycler
        if light_varname is not None:
            print(f" and {light_varname}")
            ax.plot(t, getattr(ctrl_data, light_varname) + offsets, alpha=0.5)
        title = varname
        if dashed_varname is not None:
            title += f", {dashed_varname} dashed"
        if light_varname is not None:
            title += f", {light_varname} light"
        ax.set(
            title=title,
            ylabel=varname,
        )

    axs[-1].set(xlabel="time (ms)")
    for i, (i_start, i_end) in enumerate(ctrl_data.i_trial_boundaries):
        t_start = ctrl_data.t_ms[i_start]
        axs[0].annotate(
            f"trial {i}",
            xy=(t_start, axs[0].get_ylim()[1]),
            xycoords="data",
            xytext=(0, 5),
            textcoords="offset points",
            ha="left",
            va="top",
            annotation_clip=False,
        )
        for ax in axs:
            try:
                # ax.axvline(t_start, color="xkcd:light gray", lw=0.5)
                t_end = ctrl_data.t_ms[i_end]
                ax.axvspan(t_start, t_end, color="xkcd:gray", alpha=0.1, zorder=0)
            except Exception as e:
                # probably when missing data
                pass

    plt.savefig(outfile)
    print(f"saved ctrl plots to {outfile}")
    return fig, axs


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    # setup_debugger()
    style_plots()
    expt_data = CtrlData.load(snakemake.input[0])
    plot_all(expt_data, snakemake.output[0], setting=snakemake.wildcards.setting)
    plot_all(
        expt_data,
        snakemake.output[1],
        setting=snakemake.wildcards.setting,
        explode=True,
    )
