import brian2.only as b2
import cleo
import matplotlib.pyplot as plt

from src.scripts.cleo_prep import CleoPrep
from src.plot_style import style_plots


def main(cleo_file, outfiles):
    style_plots()
    plt.switch_backend("agg")
    cleo_expt = CleoPrep.load(cleo_file)
    fig, ax = cleo.viz.plot(sim=cleo_expt.sim, scatterargs={"s": 1, "rasterized": True})
    for i_group, group in enumerate(cleo_expt.light.groups):
        for i_channel in group:
            ax.text(
                *(cleo_expt.light.coords[i_channel] / b2.um),
                f"{i_group + 1}",
                fontsize=6,
                ha="center",
            )
    for f in outfiles:
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.input[0], snakemake.output)
