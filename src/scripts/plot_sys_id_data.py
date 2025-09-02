import matplotlib.pyplot as plt

from src.data_classes import ExptData


def main(cond_params, base_params, data_file, outfiles):
    print(cond_params)
    expt_data = ExptData.load(data_file)
    n_trials = len(expt_data.Y)
    trials_per_row = 4
    rows = (n_trials + trials_per_row - 1) // trials_per_row
    fig = plt.figure(figsize=(trials_per_row * 2, 4 * rows))
    subfigs = fig.subfigures(rows, trials_per_row)

    for i, subfig in enumerate(subfigs.flatten()):
        ax_U, ax_Y = subfig.subplots(nrows=2, ncols=1)
        U, Y, t_ms = expt_data.get_trial_UYt(i)
        ax_U.plot(t_ms, U)
        ax_Y.plot(t_ms, Y)

    for f in outfiles:
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.input[0], snakemake.output)
