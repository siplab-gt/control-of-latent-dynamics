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
    return mo, np, os, plt


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
    return CtrlData, data


@app.cell
def _(CtrlData, browser, data, np, os):
    from src.helpers import load_pkl_gz

    def fsnn_test_data_from_ctrl_data(ctrl_data: CtrlData) -> dict[str, np.ndarray]:
        curr_dir = os.path.dirname(browser.value[0].id)
        while not os.path.exists(os.path.join(curr_dir, "fsnn_test_data.pkl.gz")):
            print(f"Searching for fsnn_test_data.pkl.gz in {curr_dir}")
            print(os.listdir(curr_dir))
            parent = os.path.dirname(curr_dir)
            if parent == curr_dir:  # reached root
                raise FileNotFoundError("fsnn_test_data.pkl.gz not found in parent directories")
            curr_dir = parent

        return load_pkl_gz(os.path.join(curr_dir, "fsnn_test_data.pkl.gz"))

    fsnn_test_data = fsnn_test_data_from_ctrl_data(data)
    fsnn_test_data.keys()
    return (fsnn_test_data,)


@app.cell
def _(fsnn_test_data, n_conds, n_factors, n_muscles, np):
    # get trial starts/ends and conditions from fsnn_test_data post hoc
    def get_trials_from_test_data():
        i_starts = np.where(np.diff(fsnn_test_data["in_mon"]['fin'][:, 0]) > 0)[0] + 1
        print(len(i_starts), 'trials found')
        u_task = fsnn_test_data["in_mon"]['fin'][i_starts]
        angle = np.arctan2(u_task[:, 2], u_task[:, 1])
        conds = (angle % (2 * np.pi)) / (2 * np.pi) * n_conds
        conds = np.round(conds).astype(int)
        print(counts := np.bincount(conds))
        assert np.all(counts > 0), "Some conditions have zero trials!"

        Xtest = np.zeros((len(i_starts), 81, n_factors))
        Ztest = np.zeros((len(i_starts), 81, n_muscles))
        for i_trial in range(len(i_starts)):
            cond = conds[i_trial]
            Xtest[i_trial, :, :] = fsnn_test_data["y_mon"]['y_'][i_starts[i_trial]:i_starts[i_trial]+810:10, :]
            Ztest[i_trial, :, :] = fsnn_test_data["z_mon"]['z'][i_starts[i_trial]:i_starts[i_trial]+810:10, :n_muscles]
        # fig, axs = plt.subplots(2, n_conds, figsize=(15, 5), layout='compressed')

        return conds, Xtest, Ztest

    test_conds, Xtest, Ztest = get_trials_from_test_data()
    return Xtest, Ztest, test_conds


@app.cell
def _(np, test_conds):
    np.bincount(test_conds), 
    return


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
    # ref_conds = np.linalg.norm(Zref - emg_templates[:, None, :, :], ord=2, axis=(2, 3)).argmin(
    #     axis=0
    # )
    ref_conds = data.conditions

    # start at +1 since we floored the start index
    # (trials can start any 2 ms step, while Cleo is at 10 ms steps)
    Ztrials = np.array(
        [data.Z[i_start : i_start + nt, :n_muscles] for i_start in i_starts]
    )
    Xtrials = np.array(
        [data.X[i_start : i_start + nt, :] for i_start in i_starts]
    )
    return (
        Xtrials,
        Zref,
        Ztrials,
        emg_templates,
        i_ends,
        i_starts,
        n_conds,
        n_factors,
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
def acc_pred_reach(Xtest, Xtrials, Ztest, Ztrials, ref_conds, test_conds):
    from sklearn.neighbors import KNeighborsClassifier

    def acc_pred_reach():
        results = {}
        for fit_x, fit_y, test_x, test_y, name in [
            (Xtest, test_conds, Xtrials, ref_conds, "X"),
            (Ztest, test_conds, Ztrials, ref_conds, "Z"),
        ]:
            print(f'computing accuracy for {name}...')
            classifier = KNeighborsClassifier(n_neighbors=3)
            classifier.fit(
                X=fit_x.reshape((len(fit_x), -1)),
                y=fit_y,
            )
            results[name] = classifier.score(
                X=test_x.reshape((len(test_x), -1)),
                y=test_y,
            )
        return results, classifier.predict(test_x.reshape((len(test_x), -1)))

    acc_results, pred_conds = acc_pred_reach()
    acc_results
    return (pred_conds,)


@app.cell
def _(i_ends, i_starts, np):
    np.min(i_ends - i_starts), np.max(i_ends - i_starts)
    return


@app.cell
def _(nt):
    nt
    return


@app.cell
def _(Xtest, Zref, Ztest, Ztrials, emg_templates, n_conds, plt, test_conds):
    def plot_pred_reaches():
        i = 4
        fig, axs = plt.subplots(n_conds, 4, figsize=(8, n_conds * 2), sharey='col')
        axs[0, 0].plot(Ztrials[i])
        axs[1, 0].plot(Zref[i])

        for i_cond in range(n_conds):
            ax = axs[i_cond, 1]
            ax.plot(emg_templates[i_cond])
        axs[0, 1].set(title='EMG templates')
    
        for i_test_trial in range(len(test_conds)):
            ax = axs[test_conds[i_test_trial], 2]
            ax.set_prop_cycle(None)
            ax.plot(Ztest[i_test_trial], alpha=0.2)
        axs[0, 2].set(title='Z from fSNN test data')

        for i_test_trial in range(len(test_conds)):
            ax = axs[test_conds[i_test_trial], 3]
            ax.set_prop_cycle(None)
            ax.plot(Xtest[i_test_trial], alpha=0.2)
        axs[0, 3].set(title='X from fSNN test data')

        for ax in axs.flat:
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
