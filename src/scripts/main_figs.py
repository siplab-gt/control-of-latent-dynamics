import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import CubicSpline

import src.plot_style as psty
from src.helpers import setup_debugger


def cubic_spline_interp(x, y):
    i_sort = np.argsort(x)
    assert np.all(np.diff(x[i_sort]) > 0), "x values must be unique for interpolation"
    cs = CubicSpline(x[i_sort], y[i_sort])
    return cs


seed_line_kwargs = dict(
    lw=0.4,
    estimator=None,
    units="seed",
)

mean_line_lw = 2


def main(
    cfg,
    results_csv,
    key_metrics_csv,
    outfiles,
    setting,
):
    expl_params = cfg["star_sweep_params"]
    if setting == "sufficient_stim":
        metric = "Q_track_Xref_X_fsnn"
        colorbar_label = r"$Q_\text{track}\left(X_\text{ref}, X\right)$"
        key_metric_name = "k_suff"
        main_fig_plot_type = "lineplots"
        plots_per_row = 6
    else:  # optimal_stim_rec
        assert setting == "optimal_stim_rec", f"Unknown setting: {setting}"
        metric = "Q_track_Zref_Z"
        colorbar_label = r"$Q_\text{track}\left(Z_\text{ref}, Z\right)$"
        key_metric_name = "optimal_alpha_stim"
        main_fig_plot_type = "lineplots"
        expl_params |= cfg["star_sweep_params_measurement_only"]
        plots_per_row = 7

    psty.style_plots()
    df = pd.read_csv(results_csv)
    df["alpha_stim"] = df.k / (df.m + df.k)
    key_metrics_df = pd.read_csv(key_metrics_csv)

    # Create a combined key for hue
    df["ctrl_latency"] = df["ctrl_type"] + df["latency"].astype(str)
    df["ctrl_latency"] = df["ctrl_latency"].map(
        {"OLFalse": "OL", "LQMPCFalse": "MPC", "LQMPCTrue": "MPC with latency"}
    )
    rows = int(np.ceil(len(expl_params) / plots_per_row))

    def grid_subplots(extra_row=False):
        n_rows = rows + 1 if extra_row else rows
        fig, axs = plt.subplots(
            n_rows,
            plots_per_row,
            figsize=(psty.double_col_width, 2 * n_rows),
            layout="compressed",
            sharex="col",
            sharey="row",
        )
        btm_axs = axs[1:, :] if extra_row else axs
        for i in range(len(expl_params), len(btm_axs.flatten())):
            btm_axs.flatten()[i].axis("off")
        return fig, axs

    def grid_subplots_main():
        n_rows = 1 + 2 * rows
        height_ratios = [2.0] + [1.3] * (n_rows - 1)
        fig, axs = plt.subplots(
            n_rows,
            plots_per_row,
            figsize=(psty.double_col_width, sum(height_ratios)),
            layout="compressed",
            sharex="col",
            sharey="row",
            gridspec_kw={"height_ratios": height_ratios},
        )
        gs = axs[0, 0].get_gridspec()
        ax_top = fig.add_subplot(gs[0, :])
        for ax in axs[0, :]:
            ax.remove()

        axs_key = axs[1 : 1 + rows, :]
        axs_ctrl = axs[1 + rows : 1 + 2 * rows, :]

        for axs in [axs_key, axs_ctrl]:
            for i in range(len(expl_params), len(axs.flatten())):
                axs.flatten()[i].axis("off")

        return fig, ax_top, axs_key, axs_ctrl

    main_fig, ax_top, axs_key, axs_ctrl = grid_subplots_main()

    ref_condition = cfg["reference_params"]

    # top plot
    ref_cond_except_latency = ref_condition.copy()
    del ref_cond_except_latency["latency"]
    ref_data = df.query(
        " and ".join(
            [
                f"({k} == {v!r} or {k}.isnull())"
                for k, v in ref_cond_except_latency.items()
            ]
        )
    ).copy()

    def _interp(group):
        x = group["k"].values
        y = group[metric].values
        k_interp = np.arange(x.min(), x.max() + 1)
        cs = cubic_spline_interp(x, y)
        y_interp = cs(k_interp)
        return pd.DataFrame({"k": k_interp, metric: y_interp})

    interp_ref_data = ref_data.groupby(["ctrl_latency", "seed"]).apply(
        _interp, include_groups=False
    )

    # dots to show actual data
    sns.scatterplot(
        data=ref_data,
        x="k",
        y=metric,
        hue="ctrl_latency",
        ax=ax_top,
        marker=".",
        lw=0.4,
        legend=False,
        palette=psty.ctrl_palette,
        # style="ctrl_latency",
    )
    # thin lines show interpolated per-seed data
    sns.lineplot(
        data=interp_ref_data,
        x="k",
        y=metric,
        hue="ctrl_latency",
        ax=ax_top,
        marker=None,
        errorbar=None,
        **seed_line_kwargs,
        legend=False,
        palette=psty.ctrl_palette,
        style="ctrl_latency",
        dashes=psty.ctrl_styles,
    )
    # thick lines show average of interpolated data
    sns.lineplot(
        data=interp_ref_data,
        x="k",
        y=metric,
        hue="ctrl_latency",
        ax=ax_top,
        errorbar=None,
        lw=mean_line_lw,
        palette=psty.ctrl_palette,
        style="ctrl_latency",
        dashes=psty.ctrl_styles,
    )

    def plot_lineplots(lp_axs, key_metric_name):
        """Plot lineplots showing key metric vs exploration parameter."""
        for ep, ax in zip(expl_params, lp_axs.flat):
            subset = key_metrics_df.query(
                "ep == @ep and key_metric_name == @key_metric_name"
            )
            if subset.empty:
                ax.set_visible(False)
                continue
            # per-seed lines
            sns.lineplot(
                data=subset,
                x="ep_value",
                y="key_metric_value",
                hue="ctrl_latency",
                ax=ax,
                legend=False,
                palette=psty.key_palette,
                **seed_line_kwargs,
            )
            # average
            sns.lineplot(
                data=subset,
                x="ep_value",
                y="key_metric_value",
                hue="ctrl_latency",
                ax=ax,
                legend=False,
                palette=psty.key_palette,
                errorbar=None,
                lw=mean_line_lw,
            )
            # ax.set(ylim=(0, 1))
        for ax in lp_axs[..., 1:].flat:
            ax.set(ylabel="")

    def plot_control_performance(cp_axs):
        """Plot control performance metric vs exploration parameter per seed and k."""
        for ep, ax in zip(expl_params, cp_axs.flat):
            conds2keep = ref_condition.copy()
            del conds2keep[ep], conds2keep["latency"]
            query_str = " and ".join(
                [f"({k} == {v!r} or {k}.isnull())" for k, v in conds2keep.items()]
            )
            query_str += ' and (ctrl_latency == "MPC")'
            subset = df.query(query_str)
            if subset.empty:
                ax.set_visible(False)
                continue

            k_values = np.sort(subset["k"].unique())
            k_palette = sns.color_palette("viridis", n_colors=len(k_values))
            k_palette = dict(zip(k_values, k_palette))

            sns.lineplot(
                data=subset,
                x=ep,
                y=metric,
                hue="k",
                **seed_line_kwargs,
                legend=False,
                palette=psty.cmap,
                ax=ax,
            )
            sns.lineplot(
                data=subset,
                x=ep,
                y=metric,
                hue="k",
                errorbar=None,
                lw=mean_line_lw,
                legend=(ax == cp_axs.flat[-1]),
                palette=psty.cmap,
                ax=ax,
            )

        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        for ax in cp_axs[..., 1:].flat:
            ax.set(ylabel="")

    def plot_heatmaps(hm_fig, hm_axs, mean_or_std):
        vmin, vmax = df[metric].min(), df[metric].max()
        vmin = max(vmin, -1)
        vmax = min(vmax, 1)
        heatmap_kwargs = dict(
            mean=dict(
                vmin=vmin,
                vmax=vmax,
                center=0,
                cmap=psty.div_cmap,
            ),
            std=dict(cmap=psty.cmap),
        )[mean_or_std]
        for ep, ax in zip(expl_params, hm_axs.flat):
            conds2keep = ref_condition.copy()
            del conds2keep[ep]
            query_str = " and ".join(
                [f"({k} == {v!r} or {k}.isnull())" for k, v in conds2keep.items()]
            )
            subset = df.query(query_str)
            # average over seeds
            subset = subset.groupby([ep, "k"])[[metric]]
            subset = getattr(subset, mean_or_std)().reset_index()
            subset = subset.pivot(index="k", columns=ep, values=metric)
            sns.heatmap(
                subset,
                ax=ax,
                cbar=False,
                **heatmap_kwargs,
            )
            ax.invert_yaxis()
            ax.set_title(ep)

            # Add text annotations for extreme values not represented by colormap
            for i, k_val in enumerate(subset.index):
                for j, ep_val in enumerate(subset.columns):
                    val = subset.iloc[i, j]
                    if val < vmin:
                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=8,
                        )

        for ax in hm_axs[..., 1:].flat:
            ax.set(ylabel="")

        hm_fig.colorbar(
            hm_axs.flat[0].collections[0],
            ax=hm_axs,
            label=mean_or_std + " " + colorbar_label,
        )

    # main fig: top row (Q over k), key metric row, control performance row
    if main_fig_plot_type == "lineplots":
        plot_lineplots(axs_key, key_metric_name)
    else:  # heatmaps
        plot_heatmaps(main_fig, axs_key, "mean")

    plot_control_performance(axs_ctrl)

    for ax in axs_key.flat:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", which="both", length=0)
    for axs in [axs_key, axs_ctrl]:
        for ax in axs.flat:
            ax.grid(True, color="0.9", linewidth=0.6)
        for ax in axs[:, 1:].flat:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", which="both", length=0)

    fig_lp, axs_lp = grid_subplots()
    plot_lineplots(axs_lp, key_metric_name)

    fig_hm_mean, axs_hm_mean = grid_subplots()
    plot_heatmaps(fig_hm_mean, axs_hm_mean, "mean")

    fig_hm_std, axs_hm_std = grid_subplots()
    plot_heatmaps(fig_hm_std, axs_hm_std, "std")

    for fig, files in [
        (main_fig, outfiles.main),
        (fig_lp, outfiles.lineplots),
        (fig_hm_mean, outfiles.heatmaps_mean),
        (fig_hm_std, outfiles.heatmaps_std),
    ]:
        for f in files:
            fig.savefig(f)
            print(f"Saving {f}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config.get("debug", False))

    main(
        snakemake.config,
        snakemake.input[0],
        snakemake.input[1],
        snakemake.output,
        setting=snakemake.wildcards.setting,
    )


if __name__ == "__main__" and "snakemake" not in locals():
    import argparse
    import pathlib

    import snakemake

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-csv",
        required=True,
        help="CSV file with results to plot",
    )
    parser.add_argument(
        "--key-metrics-csv",
        required=True,
        help="CSV file with precomputed key metrics",
    )
    args = parser.parse_args()

    setting_task = pathlib.Path(args.results_csv).stem
    setting = "_".join(setting_task.split("_")[:-1])
    print(f"{setting_task=}")

    exts = [".webp", ".pdf"]

    add_exts = lambda base: [f"{base}{ext}" for ext in exts]

    outfiles = argparse.Namespace(
        main=add_exts(f"results/figs/{setting_task}"),
        lineplots=add_exts(f"results/figs/{setting_task}_lineplots"),
        heatmaps_mean=add_exts(f"results/figs/{setting_task}_heatmaps_mean"),
        heatmaps_std=add_exts(f"results/figs/{setting_task}_heatmaps_std"),
    )
    print(f"{outfiles=}")

    cfg = snakemake.common.configfile._load_configfile("./config/config.yaml")

    main(cfg, args.results_csv, args.key_metrics_csv, outfiles, setting=setting)
