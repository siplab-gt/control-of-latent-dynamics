from src.plot_style import style_plots
from src.scripts.model_fit import ModelFitPredData
from src.scripts.plot_expt_data import plot_all
from src.scripts.sys_id_expts import ExptData


def plot_model_fit_predictions(mfpd: ModelFitPredData, expt_data: ExptData, outfiles):
    style_plots()
    fig, axs = plot_all(expt_data, None)
    # plot predictions on top of previous plots
    # predict true factors X from BRAID's inferred X
    axs[0].plot(mfpd.t_ms, mfpd.Xhat_lds2fsnn, linestyle="--")
    axs[1].plot(mfpd.t_ms, mfpd.Zhat, linestyle="--")
    axs[2].plot(mfpd.t_ms, mfpd.Yhat, linestyle="--")

    for f in outfiles:
        print(f"Saving figure to {f}")
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    model_fit_pred_data = ModelFitPredData.load(snakemake.input[0])
    expt_data = ExptData.load(snakemake.input[1])

    plot_model_fit_predictions(model_fit_pred_data, expt_data, snakemake.output)
