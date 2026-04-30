import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns

import src.plot_style as psty
from src.helpers import preprocess_params, setup_debugger
from src.scripts.aggregate_dependence_analysis import CMIEstimateSet
from src.scripts.key_metrics_table import interpolator

seed_line_kwargs = dict(
    lw=0.4,
    alpha=0.3,
    estimator=None,
    units="seed",
)

mean_line_lw = 1.5

# rename dict for various columns for improved display
_r = dict(
    Q_track_Xref_Xhat_lds=r"$Q_{\text{track}}(X_{\text{ref}}, \widehat{X}_{\text{LDS}})$",
    Q_track_Xref_X_fsnn=r"$Q_{\text{track}}(X_{\text{ref}}, X_{\text{fSNN}})$",
    Q_track_Zref_Z=r"$Q_{\text{track}}(Z_{\text{ref}}, Z)$",
    optimal_alpha_stim=r"$\alpha_\text{stim}^*$",
    k_knee=r"$k_{\text{knee}}$",
    k=r"$k$",
    m=r"$m$",
    n_trials_sys_id=r"$n_{\text{sys ID trials}}$",
    n_latent_dims_offset=r"$n_x$ offset",
    beam_width_um="beam width\n(μm)",
    network_area_scale="network\narea scale",
    density_scale="density\nscale",
    model_task_inputs="model\ntask inputs",
    ctrl_latency="control type",
)

_r["k_0.9"] = r"$k_{0.9}$"


def r(label):
    return _r.get(label, label)


def prettify_column_names(df):
    return df.rename(columns={col: r(col) for col in df.columns})


def main(
    base_params,
    cfg,
    setting,
    results_csv,
    key_metrics_csv,
    dependence_key_file,
    dependence_control_file,
    outfiles,
):
    expl_params = cfg["star_sweep_params"] | cfg[f"star_sweep_params_{setting}"]
    ctrl_metric = base_params.ctrl_metric
    key_metric = base_params.key_metrics[0]
    if setting == "sufficient_stim":
        colorbar_label = r"$Q_\text{track}\left(X_\text{ref}, X\right)$"
        main_fig_plot_type = "lineplots"
        plots_per_row = 6
    else:  # optimal_stim_rec
        assert setting == "optimal_stim_rec", f"Unknown setting: {setting}"
        colorbar_label = r"$Q_\text{track}\left(Z_\text{ref}, Z\right)$"
        main_fig_plot_type = "lineplots"
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

    rdf = prettify_column_names(df)
    r_key_metrics_df = prettify_column_names(key_metrics_df)

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
        n_rows = 1 + 3 * rows
        height_ratios = [2.0] + [1.3] * (n_rows - 1)
        # Using sharex=False to avoid squishing top plots with dependence plots
        fig, axs = plt.subplots(
            n_rows,
            plots_per_row,
            figsize=(psty.double_col_width, sum(height_ratios)),
            layout="compressed",
            sharex=False,
            sharey="row",
            gridspec_kw={"height_ratios": height_ratios},
        )
        gs = axs[0, 0].get_gridspec()
        ax_top = fig.add_subplot(gs[0, :])
        for ax in axs[0, :]:
            ax.remove()

        axs_key = axs[1 : 1 + rows, :]
        axs_ctrl = axs[1 + rows : 1 + 2 * rows, :]
        axs_dependence = axs[1 + 2 * rows : 1 + 3 * rows, :]

        # Manually share x-axis between Key and Control rows only
        for ax_k, ax_c in zip(axs_key.flat, axs_ctrl.flat):
            ax_k.sharex(ax_c)
            ax_k.label_outer()
            ax_k.spines["bottom"].set_visible(False)
            ax_k.tick_params(axis="x", which="both", length=0)

        for axs in [axs_key, axs_ctrl, axs_dependence]:
            for i in range(len(expl_params), len(axs.flat)):
                assert False
                axs.flatten()[i].axis("off")

        return fig, ax_top, axs_key, axs_ctrl, axs_dependence

    main_fig, ax_top, axs_key, axs_ctrl, axs_dependence = grid_subplots_main()
    main_fig.get_layout_engine().set(wspace=0.0, w_pad=0.015)

    # add panel labels
    for i, ax in enumerate(
        [ax_top, axs_key.flat[0], axs_ctrl.flat[0], axs_dependence.flat[0]]
    ):
        ax.text(
            0.015,
            1.1,
            f"{chr(ord('A') + i)}",
            transform=mtransforms.blended_transform_factory(
                main_fig.transFigure, ax.transAxes
            ),
            fontsize=plt.rcParams["figure.titlesize"],
            fontweight="bold",
        )

    ref_condition = cfg["reference_params"]
    ref_condition |= cfg[f"reference_params_{setting}"]
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
        x = group[r("k")].values
        y = group[r(ctrl_metric)].values
        k_interp = np.arange(x.min(), x.max() + 1)
        interp = interpolator(x, y)
        y_interp = interp(k_interp)
        return pd.DataFrame({r("k"): k_interp, r(ctrl_metric): y_interp})

    interp_ref_data = (
        prettify_column_names(ref_data)
        .groupby([r("ctrl_latency"), "seed"])
        .apply(_interp, include_groups=False)
    )

    # # dots to show actual data
    # sns.scatterplot(
    #     data=ref_data,
    #     x="k",
    #     y=metric,
    #     hue="ctrl_latency",
    #     ax=ax_top,
    #     marker=".",
    #     lw=0.4,
    #     legend=False,
    #     palette=psty.ctrl_palette,
    #     # style="ctrl_latency",
    # )
    # thin lines show interpolated per-seed data
    sns.lineplot(
        data=interp_ref_data,
        x=r("k"),
        y=r(ctrl_metric),
        hue=r("ctrl_latency"),
        ax=ax_top,
        marker=None,
        errorbar=None,
        **seed_line_kwargs,
        legend=False,
        palette=psty.ctrl_palette,
        style=r("ctrl_latency"),
        dashes=psty.ctrl_styles,
    )
    # thick lines show average of interpolated data
    sns.lineplot(
        data=interp_ref_data,
        x=r("k"),
        y=r(ctrl_metric),
        hue=r("ctrl_latency"),
        ax=ax_top,
        errorbar=None,
        lw=mean_line_lw,
        palette=psty.ctrl_palette,
        style=r("ctrl_latency"),
        dashes=psty.ctrl_styles,
    )
    ybot = max(interp_ref_data[r(ctrl_metric)].min() * 1.1, -1)

    ax_top.set(ylim=(ybot, 1))
    ax_top.axhline(0, color="black", lw=0.5, linestyle="--")

    def plot_lineplots(lp_axs, key_metric_name):
        """Plot lineplots showing key metric vs exploration parameter."""
        for ep, ax in zip(expl_params, lp_axs.flat):
            subset = key_metrics_df.query("ep == @ep")

            if ep not in ["signals", "model_task_inputs"]:
                subset.loc[:, "ep_value"] = pd.to_numeric(subset["ep_value"])

            if subset.empty:
                ax.set_visible(False)
                continue
            subset = prettify_column_names(subset)
            # per-seed lines
            sns.lineplot(
                data=subset,
                x="ep_value",
                y=r(key_metric),
                hue=r("ctrl_latency"),
                ax=ax,
                legend=False,
                palette=psty.key_palette,
                **seed_line_kwargs,
            )
            # average
            sns.lineplot(
                data=subset,
                x="ep_value",
                y=r(key_metric),
                hue=r("ctrl_latency"),
                ax=ax,
                legend=False,
                palette=psty.key_palette,
                errorbar=None,
                lw=mean_line_lw,
            )
            # ax.set(ylim=(0, 1))
        # for ax in lp_axs[..., 1:].flat:
        #     ax.set(ylabel="")

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
            subset = prettify_column_names(subset)

            sns.lineplot(
                data=subset,
                x=r(ep),
                y=r(ctrl_metric),
                hue=r("k"),
                **seed_line_kwargs,
                legend=False,
                palette=psty.cmap,
                ax=ax,
            )
            sns.lineplot(
                data=subset,
                x=r(ep),
                y=r(ctrl_metric),
                hue=r("k"),
                errorbar=None,
                lw=mean_line_lw,
                legend=(ax == cp_axs.flat[-1]),
                palette=psty.cmap,
                ax=ax,
            )

        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        # for ax in cp_axs[..., 1:].flat:
        # ax.set(ylabel="")
        cp_axs.flat[0].set(ylim=(-1, 1))

    def plot_heatmaps(hm_fig, hm_axs, mean_or_std):
        vmin, vmax = df[ctrl_metric].min(), df[ctrl_metric].max()
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
            subset = subset.groupby([ep, "k"])[[ctrl_metric]]
            subset = getattr(subset, mean_or_std)().reset_index()
            subset = subset.pivot(index="k", columns=ep, values=ctrl_metric)
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

    def plot_dependence(dependence_axs):
        dependence_key_res = CMIEstimateSet.load(dependence_key_file)
        dependence_ctrl_res = CMIEstimateSet.load(dependence_control_file)

        # Maps
        key_map = {res.ep: res for res in dependence_key_res.null_corrected_ep_results}
        ctrl_map = {
            res.ep: res for res in dependence_ctrl_res.null_corrected_ep_results
        }

        def get_p_crit(results_list, alpha=0.05):
            significant = [r for r in results_list if r.q_value <= alpha]
            if not significant:
                return alpha / len(results_list) if results_list else 0.05
            return max(r.p_value for r in significant)

        # We compute distinct thresholds for Key and Control analyses
        p_crit_key = get_p_crit(dependence_key_res.ep_results)
        p_crit_ctrl = get_p_crit(dependence_ctrl_res.ep_results)

        print(f"FDR Critical P: Key={p_crit_key:.5f}, Ctrl={p_crit_ctrl:.5f}")

        y_sig_marker = np.max(
            [
                np.max(r.boot_dist)
                for r in dependence_key_res.null_corrected_ep_results
                + dependence_ctrl_res.null_corrected_ep_results
            ]
        )

        def add_significance_marker(ax, res, x_marker):
            marker = ""
            if res.q_value <= 0.05:
                marker = "*"
            if res.q_value <= 0.01:
                marker = "**"
            if res.q_value <= 0.001:
                marker = "***"
            if marker:
                ax.text(
                    x_marker,
                    y_sig_marker,
                    marker,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="k",
                )

        for ep, ax in zip(expl_params, dependence_axs.flat):
            # If neither exist, hide
            if ep not in key_map and ep not in ctrl_map:
                ax.set_visible(False)
                continue

            # Prepare data for seaborn
            # We want 2 split violins side-by-side
            # X-axis: 0 -> Key, 1 -> Control
            # Hue: Null vs Boot

            plot_data = []

            # Key Data
            if ep in key_map:
                res = key_map[ep]
                n_null = len(res.null_dist)
                plot_data.extend(
                    [
                        {"Type": "Key", "Condition": "Null", "dependence": val}
                        for val in res.null_dist
                    ]
                )
                # Boot
                n_boot = len(res.boot_dist)
                plot_data.extend(
                    [
                        {"Type": "Key", "Condition": "Boot", "dependence": val}
                        for val in res.boot_dist
                    ]
                )

                # Significance threshold line value (from Null distribution)
                # If p_crit is X, we want the (1-X) percentile of Null
                thresh_key = np.percentile(res.null_dist, 100 * (1 - p_crit_key))
            else:
                thresh_key = None

            # Control Data (Note: naming "Control" to distinct from "Key")
            # But we want specific colors.
            # "Control" metric usually corresponds to `c["ctrl"]` (teal)
            # "Key" metric usually corresponds to `c["key"]` (blue/purple a1)

            if ep in ctrl_map:
                res = ctrl_map[ep]
                n_null = len(res.null_dist)
                plot_data.extend(
                    [
                        {"Type": "Control", "Condition": "Null", "cmi": val}
                        for val in res.null_dist
                    ]
                )
                # Boot
                n_boot = len(res.boot_dist)
                plot_data.extend(
                    [
                        {"Type": "Control", "Condition": "Boot", "cmi": val}
                        for val in res.boot_dist
                    ]
                )

                thresh_ctrl = np.percentile(res.null_dist, 100 * (1 - p_crit_ctrl))
            else:
                thresh_ctrl = None

            df_plot = pd.DataFrame(plot_data)

            if df_plot.empty:
                ax.set_visible(False)
                continue

            # Define palette matching hue
            # We have hue="Condition" (Null, Boot)
            # But we want different colors for Key-Boot vs Control-Boot.
            # Seaborn split violin doesn't support varying hue colors by x-group easily.
            # Workaround: Plot them separately.

            # Plot Key
            if ep in key_map:
                res = key_map[ep]
                sns.violinplot(
                    data=df_plot[df_plot["Type"] == "Key"],
                    x="Type",
                    y="dependence",
                    hue="Condition",
                    split=True,
                    inner=None,
                    palette={"Null": "0.6", "Boot": psty.c["key"]},
                    ax=ax,
                    width=0.8,
                    linewidth=0,
                    order=["Key", "Control"],  # Force x-positions
                    hue_order=["Null", "Boot"],
                    legend=False,
                )
                # Add threshold line
                # x=0. The violin is centered at 0.
                # It spans [-0.4, 0.4] roughly with width=0.8
                # Null is left side (-).
                xmin, xmax = -0.4, 0.4
                ax.hlines(
                    thresh_key, xmin, xmax, colors="gray", linestyle="--", linewidth=1.0
                )  # , linestyle="--")

                # Add dependence estimate and CI markers on the right (Boot) side
                # Key is at x=0. Boot is right side (0 to 0.4).
                # Place marker exactly at 0 (flat edge)
                x_marker = 0.0
                ax.plot(
                    [x_marker, x_marker],
                    [res.ci_low, res.ci_high],
                    color="k",
                    linewidth=2,
                )
                ax.plot(x_marker, res.cmi, color="k", marker="o", markersize=3)

                add_significance_marker(ax, res, x_marker)

            # Plot Control
            if ep in ctrl_map:
                res = ctrl_map[ep]
                sns.violinplot(
                    data=df_plot[df_plot["Type"] == "Control"],
                    x="Type",
                    y="cmi",
                    hue="Condition",
                    split=True,
                    inner=None,
                    palette={"Null": "0.6", "Boot": psty.c["ctrl"]},
                    ax=ax,
                    width=0.8,
                    linewidth=0,
                    order=["Key", "Control"],
                    hue_order=["Null", "Boot"],
                    legend=False,
                )
                # x=1.
                xmin, xmax = 0.6, 1.4  # Left side of the second violin
                ax.hlines(
                    thresh_ctrl,
                    xmin,
                    xmax,
                    colors="gray",
                    linestyle="--",
                    linewidth=1.0,
                )

                # Add dependence estimate and CI markers on the right (Boot) side
                # Control is at x=1. Boot is right side (1 to 1.4).
                x_marker = 1.0
                ax.plot(
                    [x_marker, x_marker],
                    [res.ci_low, res.ci_high],
                    color="k",
                    linewidth=2,
                )
                ax.plot(x_marker, res.cmi, color="k", marker="o", markersize=3)

                add_significance_marker(ax, res, x_marker)

            # Clean up
            ax.set_xlabel("")
            # remove x spine, ticks, labels
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
            if ax == dependence_axs.flat[0]:
                ax.set_ylabel("Null-corrected\nCMI (bits)")
            else:
                ax.set_ylabel("")
            # Or keep them? Space is tight.
            # The colors distinguish them (Purple vs Teal).
            # Maybe unnecessary text.

            # Add a legend only on the first plot?
            # Or no legend, rely on caption.

        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        handles = [
            mpatches.Patch(color="0.6", label="shuffled null"),
            mpatches.Patch(color=psty.c["key"], label=r(key_metric)),
            mpatches.Patch(color=psty.c["ctrl"], label=r(ctrl_metric)),
            mlines.Line2D(
                [],
                [],
                color="gray",
                linestyle="--",
                linewidth=1.0,
                label=r"$q=0.05$",
            ),
            mlines.Line2D(
                [],
                [],
                color="k",
                linewidth=2,
                marker="o",
                markersize=3,
                label="estimate & 95% CI",
            ),
            mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker="*",
                color="k",
                markersize=4,
                label=r"$q < 0.05$",
            ),
        ]

        fig = dependence_axs.flat[0].figure

        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            frameon=True,
        )

    if dependence_key_file is not None:
        assert dependence_control_file is not None
        plot_dependence(axs_dependence)

    # main fig: top row (Q over k), key metric row, control performance row
    if main_fig_plot_type == "lineplots":
        plot_lineplots(axs_key, key_metric)
    else:  # heatmaps
        plot_heatmaps(main_fig, axs_key, "mean")

    plot_control_performance(axs_ctrl)

    for axs in [axs_key, axs_ctrl, axs_dependence]:
        for ax in axs.flat:
            ax.grid(True, color="0.9", linewidth=0.6)
            ax.set_axisbelow(True)
        for ax in axs[:, 1:].flat:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", which="both", length=0)

    # fig_lp, axs_lp = grid_subplots()
    # plot_lineplots(axs_lp, key_metric)

    # fig_hm_mean, axs_hm_mean = grid_subplots()
    # plot_heatmaps(fig_hm_mean, axs_hm_mean, "mean")

    # fig_hm_std, axs_hm_std = grid_subplots()
    # plot_heatmaps(fig_hm_std, axs_hm_std, "std")

    for fig, files in [
        (main_fig, outfiles.main),
        # (fig_lp, outfiles.lineplots),
        # (fig_hm_mean, outfiles.heatmaps_mean),
        # (fig_hm_std, outfiles.heatmaps_std),
    ]:
        for f in files:
            fig.savefig(f)
            print(f"Saving {f}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config.get("debug", False))

    main(
        preprocess_params(snakemake.params),
        snakemake.config,
        snakemake.wildcards.setting,
        snakemake.input[0],
        snakemake.input[1],
        snakemake.input.dependence_key,
        snakemake.input.dependence_control,
        snakemake.output,
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
        "--dependence-key",
        default=None,
        help="Pickle file with dependence analysis results for key metric",
    )
    parser.add_argument(
        "--dependence-control",
        default=None,
        help="Pickle file with dependence analysis results for control metric",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
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
    setup_debugger(args.debug)

    key_metrics_csv = args.results_csv.replace(".csv", "_key_metrics.csv")

    main(
        base_params=preprocess_params(cfg[f"analysis_{setting}"]),
        cfg=cfg,
        setting=setting,
        results_csv=args.results_csv,
        key_metrics_csv=key_metrics_csv,
        dependence_key_file=args.dependence_key,
        dependence_control_file=args.dependence_control,
        outfiles=outfiles,
    )
