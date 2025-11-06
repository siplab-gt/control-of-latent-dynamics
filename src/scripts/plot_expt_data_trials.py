import matplotlib.pyplot as plt

from src.scripts.sys_id_expts import ExptData


def plot_expt_data_trials(expt_data: ExptData, outfiles):
    n_trials = len(expt_data.i_trial_boundaries)
    print(f"{n_trials=}")
    trials_per_row = 4
    rows = (n_trials + trials_per_row - 1) // trials_per_row
    fig = plt.figure(figsize=(trials_per_row * 2, 4 * rows))
    subfigs = fig.subfigures(rows, 1)
    if rows == 1:
        subfigs = [subfigs]

    for row in range(rows):
        subfig = subfigs[row]
        axs_X, axs_U = subfig.subplots(2, trials_per_row, sharex="col", sharey="row")
        axs_X[0].set(ylabel="X (a.u.)")
        axs_U[0].set(ylabel="U (mW/mm²)")
        for col in range(trials_per_row):
            i = row * trials_per_row + col
            if i >= n_trials:
                break
            print(f"plotting sys ID trial {i}")
            U, X, t_ms = expt_data.get_trial_UXt(i)
            axs_U[col].plot(t_ms, U)
            axs_X[col].plot(t_ms, X)
            axs_X[col].set(title=f"trial {i}")
            axs_U[col].set(xlabel="t (ms)")

    for f in outfiles:
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    expt_data = ExptData.load(snakemake.input[0])
    plot_expt_data_trials(expt_data, snakemake.output)
