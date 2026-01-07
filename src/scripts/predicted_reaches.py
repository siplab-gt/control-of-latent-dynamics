import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import os

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
    n_factors = data.X.shape[1]

    # predict reaches instead from Zref since Utask has direction input removed
    Zref = np.array(
        [data.Zref[i_start : i_start + nt, :n_muscles] for i_start in i_starts]
    )
    Xref = np.array(
        [data.Xref_fsnn[i_start : i_start + nt, :] for i_start in i_starts]
    ) if data.Xref_fsnn is not None else None
    # ref_conds = np.linalg.norm(Zref - emg_templates[:, None, :, :], ord=2, axis=(2, 3)).argmin(
    #     axis=0
    # )
    ref_conds = np.array(data.conditions)

    # start at +1 since we floored the start index
    # (trials can start any 2 ms step, while Cleo is at 10 ms steps)
    Ztrials = np.array(
        [data.Z[i_start : i_start + nt, :n_muscles] for i_start in i_starts]
    )
    Xtrials = np.array(
        [data.X[i_start : i_start + nt, :] for i_start in i_starts]
    )
    return (
        Xref,
        Xtrials,
        Zref,
        Ztrials,
        emg_templates,
        i_ends,
        i_starts,
        n_conds,
        n_muscles,
        nt,
        ref_conds,
    )


@app.cell
def _(Ztrials, data, emg_templates, n_conds, n_muscles, np, nt, ref_conds):
    def acc_pred_reach_old():
        assert Ztrials.shape == (data.n_trials, nt, n_muscles)
        dist = Ztrials[None, :, :, :] - emg_templates[:, None, :, :]
        assert dist.shape == (n_conds, data.n_trials, nt, n_muscles)
        dist = np.linalg.norm(dist, ord=2, axis=(2, 3))  # conds x trials
        assert dist.shape == (n_conds, data.n_trials)
        pred_conds = np.argmin(dist, axis=0)
        assert pred_conds.shape == (data.n_trials,)
        return np.mean(pred_conds == ref_conds)

    acc_pred_reach_old()
    return


@app.cell
def acc_pred_reach(Xref, Xtrials, np, ref_conds):
    from sklearn.neighbors import KNeighborsClassifier

    def acc_pred_reach(refs, true_conds, outputs, n_neighbors=3):
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        assert n_neighbors <= np.min(np.bincount(true_conds)), np.bincount(true_conds)
        classifier.fit(
            X=refs.reshape((len(refs), -1)),
            y=true_conds,
        )
        acc = classifier.score(
            X=outputs.reshape((len(outputs), -1)),
            y=true_conds,
        )
        return acc, classifier.predict(outputs.reshape((len(outputs), -1)))

    if Xref is not None:
        acc_X, pred_conds_X = acc_pred_reach(Xref, ref_conds, Xtrials)
        acc_X
    return acc_pred_reach, pred_conds_X


@app.cell
def _(Zref, Ztrials, acc_pred_reach, ref_conds):
    acc_Z, pred_conds_Z = acc_pred_reach(Zref, ref_conds, Ztrials)
    acc_Z
    return (pred_conds_Z,)


@app.cell
def _(i_ends, i_starts, np):
    np.min(i_ends - i_starts), np.max(i_ends - i_starts)
    return


@app.cell
def _(nt):
    nt
    return


@app.cell
def _(Xref, Zref, Ztrials, emg_templates, n_conds, plt, ref_conds):
    def plot_pred_reaches():
        i = 4
        fig, axs = plt.subplots(n_conds, 4, figsize=(8, n_conds * 2), sharey='col')
        for ax in axs.flat:
            ax.axis("off")
        axs[0, 0].plot(Ztrials[i])
        axs[1, 0].plot(Zref[i])

        for i_cond in range(n_conds):
            ax = axs[i_cond, 1]
            ax.plot(emg_templates[i_cond])
        axs[0, 1].set(title='EMG templates')
    
        for i_trial in range(len(ref_conds)):
            ax = axs[ref_conds[i_trial], 2]
            ax.set_prop_cycle(None)
            ax.plot(Zref[i_trial], alpha=0.2)
        axs[0, 2].set(title='Zref from ctrl data')

        if Xref is None:
            return fig

        for i_trial in range(len(ref_conds)):
            ax = axs[ref_conds[i_trial], 3]
            ax.set_prop_cycle(None)
            ax.plot(Xref[i_trial], alpha=0.2)
        axs[0, 3].set(title='Xref from ctrl data')

        return fig

    plot_pred_reaches()
    return


@app.cell
def _(pred_conds_Z, ref_conds):
    import pandas as pd

    pd.DataFrame(
        {
            "ref_cond": ref_conds,
            "pred_cond": pred_conds_Z,
        }
    )
    return


@app.cell
def _(plt, pred_conds_Z, ref_conds):
    plt.plot(ref_conds, pred_conds_Z, "o")
    return


@app.cell
def _(Xref, plt, pred_conds_X, ref_conds):
    if Xref is not None:
        plt.plot(ref_conds, pred_conds_X, "o")
    return


@app.cell
def _(
    Zref,
    Ztrials,
    i_starts,
    mo,
    n_conds,
    np,
    nt,
    plt,
    pred_conds_Z,
    ref_conds,
):
    import src.plot_style as psty

    def plot_incorrect_predictions(pred_conds, refs, outputs):
        psty.style_plots("notebook")
        i2plot = np.where(pred_conds != ref_conds)[0]
        n2plot = len(i2plot)
        ncols = 4
        nrows = int(np.ceil(n2plot / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, nrows * 2))

        avg_refs = np.array(
            [
                np.mean(refs[ref_conds == i_cond], axis=0)
                for i_cond in range(n_conds)
            ]
        )

        t = np.arange(nt) * 10  # ms
        for i_trial, ax in zip(i2plot, axs.flatten()):
            i_start = i_starts[i_trial]
            ax.axis("off")
            l1 = ax.plot(
                t,
                outputs[i_trial],
                lw=2
            )
            ax.set_prop_cycle(None)
            l2 = ax.plot(
                t,
                avg_refs[pred_conds[i_trial]],
                ls="--",
                alpha=0.5,
                lw=2
            )
            ax.set_prop_cycle(None)
            l3 = ax.plot(
                t,
                refs[i_trial],
                ls="--",
                c="k",
            )
            ax.set(title=f"trial {i_trial}")

        axs[0, 0].legend([l1[0], l2[0], l3[0]], ["actual", "predicted", "reference"])
        return mo.vstack([mo.md("## Incorrectly predicted trials"), fig])

    plot_incorrect_predictions(pred_conds_Z, Zref, Ztrials)
    return (plot_incorrect_predictions,)


@app.cell
def _(Xref, Xtrials, plot_incorrect_predictions, pred_conds_X):
    if Xref is not None:
        plot_incorrect_predictions(pred_conds_X, Xref, Xtrials)

    return


if __name__ == "__main__":
    app.run()
