import cleo
import matplotlib.pyplot as plt

from src.data_classes import CleoExperiment
from src.plot_style import style_plots


def main(cleo_file, out_file):
    style_plots()
    plt.switch_backend("agg")
    cleo_expt = CleoExperiment.load(cleo_file)
    fig, ax = cleo.viz.plot(sim=cleo_expt.sim, scatterargs={"s": 1})
    fig.savefig(out_file)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.input[0], snakemake.output[0])
