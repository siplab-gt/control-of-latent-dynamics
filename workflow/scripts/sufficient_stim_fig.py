import matplotlib.pyplot as plt
import pandas as pd

from helpers import style_plots


def main(cfg, results_csv, outfile):
    style_plots()
    df = pd.read_csv(results_csv)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 4, 8])
    ax.set(title=df["task"][0])
    fig.savefig(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.config, snakemake.input[0], snakemake.output[0])
