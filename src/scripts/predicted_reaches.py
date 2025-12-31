import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import brian2.only as b2
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    # File picker UI using marimo's file upload component
    browser = mo.ui.file_browser(
        initial_path="../../results",
        filetypes=[".gz"],
        label="Select a `ctrl_data.pkl.gz` file",
        multiple=False,
    )

    browser
    return (browser,)


@app.cell
def _(browser):
    from src.scripts.ctrl_expts import CtrlData

    data = CtrlData.load(browser.value[0].id)
    f"✅ loaded data from {browser.value[0].id} with {data.n_trials} trials"
    return (data,)


@app.cell
def _(data, np):
    from src.helpers import load_factors_emg

    # +1 since control period can start before trial really starts
    i_starts = data.i_trial_boundaries[:, 0] + 0
    i_ends = data.i_trial_boundaries[:, 1]
    # u_task = data.Utask[i_starts]
    # angle = np.arctan2(u_task[:, 1], u_task[:, 0])
    # ref_conds = angle / (2 * np.pi)

    _, emg_templates = load_factors_emg("../../src/fsnn/reaching_data.npz")
    n_conds = 8
    assert emg_templates.shape[0] == n_conds
    n_muscles = 2
    assert emg_templates.shape[2] == n_muscles
    # invert the first signal, as fSNN does
    emg_templates[..., 0] *= -1
    # EMG data is spaced by 10 ms
    # assert 10 * b2.ms == cfg.prep_fsnn.dt * cfg.cleo_prep.subsample_factor
    # if it weren't we'd need to match timescales here
    nt = emg_templates.shape[1]

    # predict reaches instead from Zref since Utask has direction input removed
    Zref = np.array(
        [data.Zref[i_start : i_start + nt, :n_muscles] for i_start in i_starts]
    )
    ref_conds = np.linalg.norm(Zref - emg_templates[:, None, :, :], ord=2, axis=(2, 3)).argmin(
        axis=0
    )

    # start at +1 since we floored the start index
    # (trials can start any 2 ms step, while Cleo is at 10 ms steps)
    Ztrials = np.array(
        [data.Z[i_start : i_start + nt, :n_muscles] for i_start in i_starts]
    )
    assert Ztrials.shape == (data.n_trials, nt, n_muscles)
    dist = Ztrials[None, :, :, :] - emg_templates[:, None, :, :]
    assert dist.shape == (n_conds, data.n_trials, nt, n_muscles)
    dist = np.linalg.norm(dist, ord=2, axis=(2, 3))  # conds x trials
    assert dist.shape == (n_conds, data.n_trials)
    pred_conds = np.argmin(dist, axis=0)
    assert pred_conds.shape == (data.n_trials,)
    acc_pred_reach = np.mean(pred_conds == ref_conds)
    acc_pred_reach
    return (
        Zref,
        Ztrials,
        emg_templates,
        i_ends,
        i_starts,
        n_conds,
        n_muscles,
        nt,
        pred_conds,
        ref_conds,
    )


@app.cell
def _(i_ends, i_starts, np):
    np.min(i_ends - i_starts), np.max(i_ends - i_starts)
    return


@app.cell
def _(nt):
    nt
    return


@app.cell
def _(data):
    dir(data)
    return


@app.cell
def _(Zref, Ztrials, emg_templates, n_conds, plt):
    def plot_pred_reaches():
        i = 0
        fig, axs = plt.subplots(n_conds, 2, figsize=(8, n_conds * 2), sharey=True)
        axs[0, 0].plot(Ztrials[i])
        axs[1, 0].plot(Zref[i])

        for i_cond in range(n_conds):
            ax = axs[i_cond, 1]
            ax.plot(emg_templates[i_cond])

        for ax in axs.flatten():
            ax.axis("off")
        return fig

    plot_pred_reaches()
    return


@app.cell
def _(pred_conds, ref_conds):
    import pandas as pd

    pd.DataFrame(
        {
            "ref_cond": ref_conds,
            "pred_cond": pred_conds,
        }
    )
    return


@app.cell
def _(plt, pred_conds, ref_conds):
    plt.plot(ref_conds, pred_conds, "o")
    return


@app.cell
def _(data, mo):
    range_slider = mo.ui.range_slider(
        start=0,
        stop=data.n_trials,
        step=1,
        value=[0, 16],
        full_width=True,
        show_value=True,
    )
    range_slider
    return (range_slider,)


@app.cell
def _(
    Ztrials,
    data,
    emg_templates,
    i_starts,
    n_muscles,
    np,
    nt,
    plt,
    pred_conds,
    range_slider,
    ref_conds,
):
    import src.plot_style as psty

    psty.style_plots("notebook")
    n2plot = range_slider.value[1] - range_slider.value[0]
    ncols = 4
    nrows = int(np.ceil(n2plot / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, nrows * 2))

    t = np.arange(nt) * 10  # ms
    for i_trial, ax in zip(
        range(range_slider.value[0], range_slider.value[1]), axs.flatten()
    ):
        i_start = i_starts[i_trial]
        ax.axis("off")
        l1 = ax.plot(
            t,
            Ztrials[i_trial],
            lw=2
        )
        ax.set_prop_cycle(None)
        l2 = ax.plot(
            t,
            emg_templates[pred_conds[i_trial], :, :],
            ls="--",
            alpha=0.5,
            lw=2
        )
        ax.set_prop_cycle(None)
        l3 = ax.plot(
            t,
            data.Zref[i_start : i_start + nt, :n_muscles],
            ls="--",
            c="k",
        )
        l4 = ax.plot(
            t,
            emg_templates[ref_conds[i_trial], :, :],
            ls=":",
            c="gray",
        )
        ax.set(title=f"trial {i_trial}")

    axs[0, 0].legend([l1[0], l2[0], l3[0], l4[0]], ["actual", "predicted", "reference", "inferred reference"])
    fig
    return


if __name__ == "__main__":
    app.run()
