import cleo
import matplotlib.pyplot as plt

from src.data_classes import fSNN
from src.plot_style import style_plots


def main(fsnn_file, out_file):
    style_plots()
    plt.switch_backend("agg")
    fsnn = fSNN.load(fsnn_file)
    fig, ax = cleo.viz.plot(fsnn.ng)
    fig.savefig(out_file)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.input[0], snakemake.output[0])
