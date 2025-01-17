import matplotlib.pyplot as plt


def main(cond_params, base_params, outfiles):
    print(cond_params)
    fig, ax = plt.subplots()
    for f in outfiles:
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.output)
