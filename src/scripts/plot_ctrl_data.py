import matplotlib.pyplot as plt
import numpy as np

from src.plot_style import style_plots
from src.scripts.ctrl_expts import CtrlData


def plot_all(ctrl_data: CtrlData, outfile, explode=False):
    vars = ["X", "Xhat_braid", "Z", "spikes", "Uopto", "Utask"]
    dashed_vars = ["Xref_fsnn", "Xref_braid", "Zref", "Yref", None, None]
    light_vars = ["Xhat_fsnn", None, "Zhat", "Yhat", None, None]
    nrows = len(vars)
    figwidth = 1 * ctrl_data.n_trials
    if not explode:
        figheight = 2 * nrows
        kwargs = {}
    elif explode:
        ht_per_series = 0.2
        num_series_per_row = [getattr(ctrl_data, v).shape[1] for v in vars]
        figheight = ht_per_series * sum(num_series_per_row)
        kwargs = {"height_ratios": num_series_per_row}
    fig, axs = plt.subplots(
        nrows, 1, figsize=(figwidth, figheight), sharex=True, **kwargs
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
        ddata = getattr(ctrl_data, dashed_varname) if dashed_varname else None
        ldata = getattr(ctrl_data, light_varname) if light_varname else None
        pctl = 90
        if explode:
            data += np.arange(data.shape[1]) * np.percentile(data, pctl)
            if ddata is not None:
                ddata += np.arange(ddata.shape[1]) * np.percentile(ddata, pctl)
            if ldata is not None:
                ldata += np.arange(ldata.shape[1]) * np.percentile(ldata, pctl)

        ax.plot(t, data)
        if dashed_varname is not None:
            ax.plot(t, ddata, ls="--")
        if light_varname is not None:
            ax.plot(t, ldata, alpha=0.5)
        ax.set(ylabel=f"{varname}, {dashed_varname} dashed, {light_varname} light")

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
            va="bottom",
            annotation_clip=False,
        )

    plt.savefig(outfile)
    return fig, axs


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    style_plots()
    expt_data = CtrlData.load(snakemake.input[0])
    plot_all(expt_data, snakemake.output[0])
    plot_all(expt_data, snakemake.output[1], explode=True)
