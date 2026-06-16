import re

import matplotlib.pyplot as plt
import numpy as np

from src.helpers import setup_debugger
from src.plot_style import c, double_col_width, style_plots
from src.scripts.ctrl_expts import CtrlData


def _representative_trial_indices(conds: np.ndarray, n_examples: int = 4) -> np.ndarray:
    unique_conds = np.unique(conds)
    chosen = []
    for c in unique_conds[:n_examples]:
        trials_for_cond = np.where(conds == c)[0]
        if len(trials_for_cond) > 0:
            chosen.append(trials_for_cond[0])
    chosen.sort()
    return np.array(chosen, dtype=int)


def _trial_slice(arr: np.ndarray, bounds: np.ndarray, i_trial: int) -> np.ndarray:
    i_start, i_end = bounds[i_trial]
    i_start = max(0, i_start - PRE_TRIAL_PADDING_SAMPLES)
    i_end = min(arr.shape[0], i_end + POST_TRIAL_PADDING_SAMPLES)
    return arr[i_start:i_end]


def _trial_slice_time(
    arr: np.ndarray, t_arr: np.ndarray, t0: float, t1: float
) -> np.ndarray:
    i0 = np.searchsorted(t_arr, t0, side="left")
    i1 = np.searchsorted(t_arr, t1, side="left")
    return arr[i0:i1]


PLOT_INDIVIDUAL_TRIALS = True
PRE_TRIAL_PADDING_SAMPLES = 0
POST_TRIAL_PADDING_SAMPLES = 0


def _cond_trial_stats(
    arr: np.ndarray, bounds: np.ndarray, conds: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for cond in np.sort(np.unique(conds)):
        idx = np.where(conds == cond)[0]
        if idx.size == 0:
            continue
        trials = [_trial_slice(arr, bounds, int(i)) for i in idx]
        min_len = min(tr.shape[0] for tr in trials)
        if min_len == 0:
            continue
        stacked = np.stack([tr[:min_len] for tr in trials], axis=0)
        mean_arr = stacked.mean(axis=0)
        sem_arr = stacked.std(axis=0) / np.sqrt(stacked.shape[0])
        out[int(cond)] = (mean_arr, sem_arr, stacked)
    return out


def _parse_km_from_path(path: str) -> tuple[str, str]:
    match = re.search(r"/k_([^/]+)/m_([^/]+)/", path)
    if match is None:
        return "?", "?"
    return match.group(1), match.group(2)


def _get_control_and_reference(
    ctrl_data: CtrlData, setting: str
) -> tuple[np.ndarray, np.ndarray, str]:
    if setting == "sufficient_stim":
        ref = ctrl_data.Xref_fsnn
        assert ref is not None and ref.size > 0, "Missing Xref_fsnn in ctrl_data"
        return ctrl_data.X, ref, "x"
    if setting == "optimal_stim_rec":
        return ctrl_data.Z[:, :2], ctrl_data.Zref[:, :2], "z"
    raise ValueError(f"Unsupported setting {setting}")


def _assert_shared_structure(ctrl_datas: list[CtrlData]):
    cond0 = np.array(ctrl_datas[0].conditions)
    bounds0 = ctrl_datas[0].i_trial_boundaries
    t0 = ctrl_datas[0].t_ms
    for i, cd in enumerate(ctrl_datas[1:], start=1):
        assert np.array_equal(cond0, np.array(cd.conditions)), (
            f"conditions differ across ctrl_data files at index {i}"
        )
        assert np.array_equal(bounds0, cd.i_trial_boundaries), (
            f"i_trial_boundaries differ across ctrl_data files at index {i}"
        )
        assert np.array_equal(t0, cd.t_ms), (
            f"t_ms differ across ctrl_data files at index {i}"
        )


def _assert_reference_consistent(ctrl_datas: list[CtrlData], atol: float = 1e-3):
    """Assert shared reference trajectory via Zref, robust to tiny fp jitter."""
    zref0 = ctrl_datas[0].Zref
    for i, cd in enumerate(ctrl_datas[1:], start=1):
        assert cd.Zref.shape == zref0.shape, (
            f"Zref shape mismatch between files: {zref0.shape} vs {cd.Zref.shape} (index {i})"
        )
        assert np.allclose(cd.Zref, zref0, atol=atol, rtol=0), (
            f"Zref mismatch between files at index {i} (max abs diff "
            f"{np.max(np.abs(cd.Zref - zref0)):.3g})"
        )


def _imshow_heat(
    ax,
    arr: np.ndarray,
    cmap: str,
    title: str = "",
    vmin=None,
    vmax=None,
    hide_axes=True,
):
    if arr.size == 0:
        ax.axis("off")
        return None
    im = ax.imshow(
        arr.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            -0.5 - PRE_TRIAL_PADDING_SAMPLES,
            arr.shape[0] - 0.5 - PRE_TRIAL_PADDING_SAMPLES,
            -0.5,
            arr.shape[1] - 0.5,
        ],
        rasterized=True,
    )
    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if title:
        ax.set_title(title)
    return im


def _plot_cond_grid(
    axs_line,
    cond_stats,
    ref_cond_stats,
    color,
    max_dims,
    show_titles=False,
    target_conds=list(range(8)),
):
    y_lims = []
    ref_lw = plt.rcParams["lines.linewidth"]
    fact_lw = plt.rcParams["lines.linewidth"]
    for ax, cond in zip(axs_line, target_conds):
        if cond not in cond_stats and cond not in ref_cond_stats:
            ax.axis("off")
            continue

        if cond in ref_cond_stats:
            ref_arr, ref_sem, ref_stacked = ref_cond_stats[cond]
            n_dims_ref = min(ref_arr.shape[1], max_dims)
            x_arr = np.arange(
                -PRE_TRIAL_PADDING_SAMPLES, ref_arr.shape[0] - PRE_TRIAL_PADDING_SAMPLES
            )
            for dim in range(n_dims_ref):
                ax.fill_between(
                    x_arr,
                    ref_arr[:, dim] - ref_sem[:, dim],
                    ref_arr[:, dim] + ref_sem[:, dim],
                    alpha=0.3,
                    color="gray",
                    zorder=0,
                    linewidth=0,
                    rasterized=True,
                )
            ax.plot(
                x_arr,
                ref_arr[:, :n_dims_ref],
                lw=ref_lw,
                ls="-",
                c="k",
                alpha=0.9,
                zorder=1,
                rasterized=True,
            )

        if cond in cond_stats:
            mean_arr, sem_arr, stacked = cond_stats[cond]
            n_dims = min(mean_arr.shape[1], max_dims)
            x_arr = np.arange(
                -PRE_TRIAL_PADDING_SAMPLES,
                mean_arr.shape[0] - PRE_TRIAL_PADDING_SAMPLES,
            )

            if PLOT_INDIVIDUAL_TRIALS:
                for dim in range(n_dims):
                    for trial_idx in range(stacked.shape[0]):
                        ax.plot(
                            x_arr,
                            stacked[trial_idx, :, dim],
                            lw=0.5,
                            ls="-",
                            c=color,
                            alpha=0.1,
                            zorder=1,
                            rasterized=True,
                        )
            else:
                for dim in range(n_dims):
                    ax.fill_between(
                        x_arr,
                        mean_arr[:, dim] - sem_arr[:, dim],
                        mean_arr[:, dim] + sem_arr[:, dim],
                        alpha=0.3,
                        color=color,
                        zorder=1,
                        linewidth=0,
                        rasterized=True,
                    )
                ax.plot(
                    x_arr,
                    mean_arr[:, :n_dims],
                    lw=fact_lw,
                    color=color,
                    alpha=0.95,
                    zorder=2,
                    rasterized=True,
                )

        if show_titles:
            inset_ax = ax.inset_axes([0.35, 0.85, 0.3, 0.3])
            inset_ax.patch.set_alpha(0.0)
            for spine in inset_ax.spines.values():
                spine.set_visible(False)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_xlim(-1.2, 1.2)
            inset_ax.set_ylim(-1.2, 1.2)
            theta = cond / 8.0 * 2 * np.pi
            dx = np.cos(theta)
            dy = np.sin(theta)
            inset_ax.arrow(
                0,
                0,
                dx,
                dy,
                width=0.1,
                head_width=0.4,
                head_length=0.4,
                fc="k",
                ec="k",
                length_includes_head=True,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if cond in ref_cond_stats:
            N = ref_cond_stats[cond][0].shape[0]
        elif cond in cond_stats:
            N = cond_stats[cond][0].shape[0]
        else:
            N = 81
        ax.set_xlim(
            -0.5 - PRE_TRIAL_PADDING_SAMPLES, N - 0.5 - PRE_TRIAL_PADDING_SAMPLES
        )
        y_lims.append(ax.get_ylim())

    if y_lims:
        y_min = min(l[0] for l in y_lims)
        y_max = max(l[1] for l in y_lims)
        for ax in axs_line:
            if ax.has_data():
                ax.set_ylim(y_min, y_max)


def make_example_fig(q1_ctrl_datas, q1_paths, q2_ctrl_datas, q2_paths):
    fig = plt.figure(figsize=(double_col_width, 5.4), layout="constrained")
    outer = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[1.2, 1.2, 1.2, 0.3],
    )

    import seaborn as sns
    from matplotlib.colors import to_rgb

    darker_opto = tuple(0.85 * val for val in to_rgb(c["opto1"]))
    opto_cmap = sns.light_palette(darker_opto, as_cmap=True)

    def add_row_label(ax, letter, caption):
        ax.text(
            -0.15,
            1.15,
            f"{letter}",
            transform=ax.transAxes,
            fontsize=plt.rcParams["figure.titlesize"],
            fontweight="bold",
            va="baseline",
            ha="right",
        )
        ax.text(
            -0.05,
            1.15,
            caption,
            transform=ax.transAxes,
            fontsize=plt.rcParams["axes.titlesize"],
            style="italic",
            va="baseline",
            ha="left",
        )

    target_conds = [0, 2, 4, 6]
    all_axes = []
    bottom_row_axes = []

    opto_vmins = [0, 0]
    opto_vmaxs = [0, 0]

    for q_idx, ctrl_datas in enumerate([q1_ctrl_datas, q2_ctrl_datas]):
        if ctrl_datas:
            opto_vmaxs[q_idx] = max(
                np.max(np.sqrt(cd.Uopto)) for cd in ctrl_datas if cd.Uopto.size > 0
            )

    for row in range(3):
        gs_row = outer[row, 0].subgridspec(
            2,
            9,
            width_ratios=[1, 1, 1, 1, 0.2, 1, 1, 1, 1],
            wspace=0.1,
            hspace=0.02,
        )

        sets = [
            ("sufficient_stim", c["x"], q1_ctrl_datas, q1_paths, ["A", "B", "C"]),
            ("optimal_stim_rec", c["z"], q2_ctrl_datas, q2_paths, ["D", "E", "F"]),
        ]

        for q_idx, (setting, base_color, ctrl_datas, input_paths, letters) in enumerate(
            sets
        ):
            cd = ctrl_datas[row]
            path = input_paths[row]
            letter = letters[row]

            achieved_and_ref = _get_control_and_reference(cd, setting)
            arr = achieved_and_ref[0]
            ref_data = achieved_and_ref[1]
            var_name = achieved_and_ref[2]

            conds = np.array(cd.conditions)
            bounds = cd.i_trial_boundaries

            ref_cd = ctrl_datas[0]
            ref_conds = np.array(ref_cd.conditions)
            bounds_ref = ref_cd.i_trial_boundaries

            k_val, m_val = _parse_km_from_path(path)
            caption = f"k={k_val}" if m_val == "NA" else f"k={k_val}, m={m_val}"

            col_offset = 0 if q_idx == 0 else 5

            axs_line = []
            for col_i, cond in enumerate(target_conds):
                if col_i == 0:
                    axs_line.append(fig.add_subplot(gs_row[0, col_i + col_offset]))
                else:
                    axs_line.append(
                        fig.add_subplot(
                            gs_row[0, col_i + col_offset],
                            sharey=axs_line[0],
                            sharex=axs_line[0],
                        )
                    )
            all_axes.extend(axs_line)

            cond_stats = _cond_trial_stats(arr, bounds, conds)
            ref_cond_stats = _cond_trial_stats(ref_data, bounds_ref, ref_conds)

            show_titles = row == 0
            _plot_cond_grid(
                axs_line,
                cond_stats,
                ref_cond_stats,
                color=base_color,
                max_dims=8,
                show_titles=show_titles,
                target_conds=target_conds,
            )

            if show_titles:
                axs_line[0].text(
                    -0.05,
                    0.48,
                    f"${var_name}_{{ref}},$",
                    color="k",
                    transform=axs_line[0].transAxes,
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=plt.rcParams["axes.labelsize"],
                )
                axs_line[0].text(
                    -0.05,
                    0.53,
                    f"${var_name}_{{fSNN}}$",
                    color=base_color,
                    transform=axs_line[0].transAxes,
                    rotation=90,
                    va="bottom",
                    ha="right",
                    fontsize=plt.rcParams["axes.labelsize"],
                )

            add_row_label(axs_line[0], letter, caption)

            # Opto heatmaps
            for col_i, cond in enumerate(target_conds):
                ax_bottom = fig.add_subplot(gs_row[1, col_i + col_offset])
                all_axes.append(ax_bottom)
                if row == 2:
                    bottom_row_axes.append(ax_bottom)
                idx = np.where(conds == cond)[0]
                if len(idx) == 0:
                    ax_bottom.axis("off")
                    continue

                opto_trials = []
                for i in idx:
                    i_start, i_end = bounds[int(i)]
                    t0 = cd.t_ms[i_start]
                    t1 = cd.t_ms[i_end - 1] if i_end > i_start else t0
                    opto_trials.append(
                        np.sqrt(_trial_slice_time(cd.Uopto, cd.t_opto_ms, t0, t1))
                    )

                min_len = min(tr.shape[0] for tr in opto_trials) if opto_trials else 0
                if min_len == 0:
                    ax_bottom.axis("off")
                    continue

                stacked_opto = np.stack([tr[:min_len] for tr in opto_trials], axis=0)
                avg_opto = stacked_opto.mean(axis=0)

                im_o = _imshow_heat(
                    ax_bottom,
                    avg_opto,
                    cmap=opto_cmap,
                    vmin=opto_vmins[q_idx],
                    vmax=opto_vmaxs[q_idx],
                    hide_axes=True,
                )
                ax_bottom.set_xlim(
                    -0.5 - PRE_TRIAL_PADDING_SAMPLES,
                    avg_opto.shape[0] - 0.5 - PRE_TRIAL_PADDING_SAMPLES,
                )

                if row == 0 and col_i == 0:
                    ax_bottom.text(
                        -0.05,
                        0.5,
                        r"$u$ channels",
                        transform=ax_bottom.transAxes,
                        rotation=90,
                        va="center",
                        ha="right",
                        fontsize=plt.rcParams["axes.labelsize"],
                    )
                if q_idx == 0 and row == 2 and col_i == 0:
                    import matplotlib.transforms as mtransforms

                    trans = mtransforms.blended_transform_factory(
                        ax_bottom.transData, ax_bottom.transAxes
                    )
                    label_size = plt.rcParams["axes.labelsize"]
                    ax_bottom.plot(
                        [0, 0, 35, 35],
                        [-0.05, 0, 0, -0.05],
                        color="k",
                        lw=1.0,
                        transform=trans,
                        clip_on=False,
                        zorder=10,
                    )
                    ax_bottom.text(
                        17.5,
                        -0.1,
                        "prep",
                        ha="center",
                        va="top",
                        transform=trans,
                        fontsize=label_size,
                        zorder=10,
                    )
                    ax_bottom.plot(
                        [35, 35, 81, 81],
                        [-0.05, 0, 0, -0.05],
                        color="k",
                        lw=1.0,
                        transform=trans,
                        clip_on=False,
                        zorder=10,
                    )
                    ax_bottom.text(
                        58,
                        -0.1,
                        "move",
                        ha="center",
                        va="top",
                        transform=trans,
                        fontsize=label_size,
                        zorder=10,
                    )

    if q1_ctrl_datas:  # just to use im_o for colorbar
        # We need a generic im_o to pass to colorbar, we can just use the last im_o
        try:
            gs_cbar = outer[3, 0].subgridspec(1, 3, width_ratios=[1, 1.2, 1])
            cax = fig.add_subplot(gs_cbar[0, 1])
            cbar = fig.colorbar(im_o, cax=cax, orientation="horizontal")
            cbar.ax.text(
                -0.04,
                0.5,
                r"$\sqrt{u}$",
                va="center",
                ha="right",
                transform=cbar.ax.transAxes,
                fontsize=plt.rcParams["axes.labelsize"],
            )
        except Exception:
            pass

    return fig


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config["debug"])
    style_plots()

    q1_ctrl_datas = [CtrlData.load(f) for f in snakemake.input.q1_files]
    q2_ctrl_datas = [CtrlData.load(f) for f in snakemake.input.q2_files]

    fig = make_example_fig(
        q1_ctrl_datas,
        snakemake.input.q1_files,
        q2_ctrl_datas,
        snakemake.input.q2_files,
    )

    plt.rcParams["savefig.bbox"] = "standard"
    for outfile in snakemake.output:
        fig.savefig(outfile)
        print(f"saved {outfile}")
