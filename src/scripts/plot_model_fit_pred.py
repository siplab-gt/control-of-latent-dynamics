import numpy as np

from src.helpers import preprocess_params, setup_debugger
from src.model_fit import LDS
from src.plot_style import style_plots
from src.scripts.model_fit import ModelFitPredData
from src.scripts.plot_expt_data_wide import plot_expt_data_wide
from src.scripts.sys_id_expts import ExptData


def plot_model_fit_predictions(
    cond_params: dict,
    mfpd: ModelFitPredData,
    expt_data: ExptData,
    lds: LDS,
    outfiles,
    style="default",
):
    style_plots()

    # Determine offsets for exploded style
    if style == "exploded":
        # Collect all data to calculate offsets
        all_data = [expt_data.X, expt_data.Z, expt_data.spikes, expt_data.lfp]
        offsets_list = []
        for data in all_data:
            if data is not None and data.size > 0:
                offset = np.arange(data.shape[1]) * np.percentile(np.ptp(data), 60)
            else:
                offset = 0
            offsets_list.append(offset)
    else:
        offsets_list = [0, 0, 0, 0]

    fig, axs = plot_expt_data_wide(expt_data, None, style=style)
    # plot predictions on top of previous plots
    for ax in axs:
        ax.set_prop_cycle(None)  # reset color cycler
    # predict true factors X from inferred X
    lw = 0.5
    ls = "-"
    if cond_params.signals in ["both", "spikes"]:
        i_split = expt_data.spikes.shape[1]
    elif cond_params.signals == "lfp":
        i_split = 0
    pred_spikes, pred_lfp = np.split(mfpd.Yhat, [i_split], axis=1)

    if style != "heatmap":
        axs[0].plot(
            mfpd.t_ms, mfpd.Xhat_lds2fsnn + offsets_list[0], c="k", ls=ls, lw=lw
        )
        axs[1].plot(mfpd.t_ms, mfpd.Zhat + offsets_list[1], c="k", ls=ls, lw=lw)
        axs[2].plot(mfpd.t_ms, pred_spikes + offsets_list[2], c="k", ls=ls, lw=lw)
        axs[3].plot(mfpd.t_ms, pred_lfp + offsets_list[3], c="k", ls=ls, lw=lw)

    U_lds = expt_data.Uopto
    if lds.k_task > 0:
        U_lds = np.hstack([U_lds, expt_data.Utask])
    U_lds = (U_lds - lds.u_mean) / lds.u_std

    dt = expt_data.t_ms[1] - expt_data.t_ms[0]
    # plot rollouts of different lengths
    if style != "heatmap":
        i_ahead_start = 0
        Xhat_ahead = (expt_data.X - lds.x_mean) / lds.x_std
        for n_ahead, alpha in [
            (5, 0.75),
            (25, 0.5),
            (50, 0.25),
        ]:
            for ax in axs:
                ax.set_prop_cycle(None)  # reset color cycler
            for i_ahead in range(i_ahead_start, n_ahead):
                Xhat_ahead = Xhat_ahead @ lds.A.T + U_lds[i_ahead] @ lds.B.T
            i_ahead_start = n_ahead

            t_ms = expt_data.t_ms + n_ahead * dt
            axs[0].plot(
                t_ms,
                Xhat_ahead * lds.x_std + lds.x_mean + offsets_list[0],
                alpha=alpha,
                ls=ls,
                lw=lw,
            )
            axs[1].plot(
                t_ms,
                lds.x2z(Xhat_ahead) * lds.z_std + lds.z_mean + offsets_list[1],
                alpha=alpha,
                ls=ls,
                lw=lw,
            )
            spikes_ahead, lfp_ahead = np.split(
                Xhat_ahead @ lds.C.T * lds.y_std + lds.y_mean, [i_split], axis=1
            )
            axs[2].plot(t_ms, spikes_ahead + offsets_list[2], alpha=alpha, ls=ls, lw=lw)
            axs[3].plot(t_ms, lfp_ahead + offsets_list[3], alpha=alpha, ls=ls, lw=lw)

    for f in outfiles:
        print(f"Saving figure to {f}")
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    model_fit_pred_data = ModelFitPredData.load(snakemake.input[0])
    expt_data = ExptData.load(snakemake.input[1])
    lds = LDS.load(snakemake.input[2])
    setup_debugger(snakemake.config["debug"])
    style = snakemake.wildcards.get("style", "default")

    plot_model_fit_predictions(
        preprocess_params(snakemake.wildcards),
        model_fit_pred_data,
        expt_data,
        lds,
        snakemake.output,
        style=style,
    )
