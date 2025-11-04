import brian2.only as b2
import cleo
import matplotlib.pyplot as plt

from src.fsnn import fSNN
from src.helpers import preprocess_params
from src.plot_style import style_plots
from src.scripts.cleo_prep import CleoPrep


def main(cond_params, base_params, outfiles):
    style_plots()
    plt.switch_backend("agg")
    fsnn = fSNN.build(cond_params, base_params)
    cleo_expt = CleoPrep.build(fsnn, cond_params, base_params)
    fig, ax = cleo.viz.plot(
        fsnn.ng_main, sim=cleo_expt.sim, scatterargs={"s": 0.1, "rasterized": True}
    )
    for i_group, group in enumerate(cleo_expt.light.groups):
        for i_channel in group:
            ax.text(
                *(cleo_expt.light.coords[i_channel] / b2.um),
                f"{i_group + 1}",
                fontsize=6,
                ha="center",
                zorder=10,
            )
    for f in outfiles:
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(
        preprocess_params(snakemake.wildcards),
        preprocess_params(snakemake.params),
        snakemake.output,
    )
