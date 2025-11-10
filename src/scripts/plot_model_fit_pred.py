import numpy as np

from src.model_fit import LDS
from src.plot_style import style_plots
from src.scripts.model_fit import ModelFitPredData
from src.scripts.plot_expt_data_wide import plot_expt_data_wide
from src.scripts.sys_id_expts import ExptData


def plot_model_fit_predictions(
    mfpd: ModelFitPredData, expt_data: ExptData, lds: LDS, outfiles
):
    style_plots()
    fig, axs = plot_expt_data_wide(expt_data, None)
    # plot predictions on top of previous plots
    for ax in axs:
        ax.set_prop_cycle(None)  # reset color cycler
    # predict true factors X from inferred X
    lw = 0.5
    ls = ":"
    axs[0].plot(mfpd.t_ms, mfpd.Xhat_lds2fsnn, c="k", ls=ls, lw=lw)
    axs[1].plot(mfpd.t_ms, mfpd.Zhat, c="k", ls=ls, lw=lw)
    axs[2].plot(mfpd.t_ms, mfpd.Yhat, c="k", ls=ls, lw=lw)

    U_lds = expt_data.Uopto
    if lds.k_task > 0:
        U_lds = np.hstack([U_lds, expt_data.Utask])
    U_lds = (U_lds - lds.u_mean) / lds.u_std

    dt = expt_data.t_ms[1] - expt_data.t_ms[0]
    # plot rollouts of different lengths
    i_ahead_start = 0
    Xhat_ahead = (expt_data.X - lds.x_mean) / lds.x_std
    for n_ahead, color in [(5, "xkcd:crimson"), (25, "xkcd:red"), (50, "xkcd:coral")]:
        for i_ahead in range(i_ahead_start, n_ahead):
            Xhat_ahead = Xhat_ahead @ lds.A.T + U_lds[i_ahead] @ lds.B.T
        i_ahead_start = n_ahead

        t_ms = expt_data.t_ms + n_ahead * dt
        axs[0].plot(t_ms, Xhat_ahead * lds.x_std + lds.x_mean, c=color, ls=ls, lw=lw)
        axs[1].plot(
            t_ms, lds.x2z(Xhat_ahead) * lds.z_std + lds.z_mean, c=color, ls=ls, lw=lw
        )
        axs[2].plot(
            t_ms, Xhat_ahead @ lds.C.T * lds.y_std + lds.y_mean, c=color, ls=ls, lw=lw
        )

    for f in outfiles:
        print(f"Saving figure to {f}")
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    model_fit_pred_data = ModelFitPredData.load(snakemake.input[0])
    expt_data = ExptData.load(snakemake.input[1])
    lds = LDS.load(snakemake.input[2])

    plot_model_fit_predictions(model_fit_pred_data, expt_data, lds, snakemake.output)
