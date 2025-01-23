import brian2.only as b2
import cleo
import matplotlib.pyplot as plt

from src.cleo_prep import cleo_prep
from src.fsnn import fSNN
from src.plot_style import style_plots


def main(fsnn_file, cond_params, config, outfiles):
    style_plots()
    plt.switch_backend("agg")
    fsnn = fSNN.load(fsnn_file)
    cleo_expt = cleo_prep(cond_params, config, fsnn)
    fig, ax = cleo.viz.plot(sim=cleo_expt.sim, scatterargs={"s": 1, "rasterized": True})
    for i_group, group in enumerate(cleo_expt.light_array.groups):
        for i_channel in group:
            ax.text(
                *(cleo_expt.light_array.coords[i_channel] / b2.um),
                f"{i_group + 1}",
                fontsize=6,
                ha="center",
            )
    for f in outfiles:
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.input[0], snakemake.wildcards, snakemake.config, snakemake.output)
