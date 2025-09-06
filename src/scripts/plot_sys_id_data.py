import matplotlib.pyplot as plt

from src.scripts.sys_id_expts import ExptData


def main(cond_params, base_params, data_file, outfiles):
    print(cond_params)
    expt_data = ExptData.load(data_file)
    n_trials = len(expt_data.i_trial_boundaries)
    print(f"{n_trials=}")
    trials_per_row = 4
    rows = (n_trials + trials_per_row - 1) // trials_per_row
    fig = plt.figure(figsize=(trials_per_row * 2, 4 * rows))
    subfigs = fig.subfigures(rows, 1)

    for row in range(rows):
        subfig = subfigs[row]
        axs_Y, axs_U = subfig.subplots(2, trials_per_row, sharex="col", sharey="row")
        axs_Y[0].set(ylabel="Y (a.u.)")
        axs_U[0].set(ylabel="U (mW/mm²)")
        for col in range(trials_per_row):
            i = row * trials_per_row + col
            if i >= n_trials:
                break
            print(f"plotting trial {i}")
            U, Y, t_ms = expt_data.get_trial_UYt(i)
            axs_U[col].plot(t_ms, U)
            axs_Y[col].plot(t_ms, Y)
            axs_Y[col].set(title=f"trial {i}")
            axs_U[col].set(xlabel="t (ms)")

    for f in outfiles:
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.input[0], snakemake.output)
