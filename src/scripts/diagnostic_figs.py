import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api import types as ptypes

import src.plot_style as psty
from src.helpers import setup_debugger

metric_groups = [
    (
        "fSNN",
        [
            "R2_fsnn_target_factors_mean",
            "R2_fsnn_target_output_mean",
        ],
        "R2",
    ),
    (
        "fSNN n_factors",
        [
            "n_factors",
        ],
        "# factors",
    ),
    (
        "Model fit",
        [
            "R2_model_fit_X_Xhat_fsnn",
            "R2_model_fit_Z_Zhat",
            "R2_model_fit_Z_X",
            # "R2_model_fit_Y_Yhat",
            "largest_eigval",
        ],
        "R2",
    ),
    (
        "Control",
        [
            "Q_track_Zref_Z",
            "Q_track_Xref_X_fsnn",
            "acc_pred_cond_X",
            "acc_pred_cond_Z",
        ],
        "Q",
    ),
    (
        "Latency",
        [
            "ctrl_compute_latency_mean_ms",
            "ctrl_compute_latency_nonsolve_ms",
        ],
        "latency (ms)",
    ),
]

metric_labels = {
    "R2_fsnn_target_factors_mean": "fSNN R2(X)",
    "R2_fsnn_target_output_mean": "fSNN R2(Z)",
    "R2_model_fit_X_Xhat_fsnn": "fit R2(X)",
    "R2_model_fit_Z_Zhat": "fit R2(Z)",
    "R2_model_fit_Z_X": "fit R2(Z|X)",
    "R2_model_fit_Y_Yhat": "fit R2(Y)",
    "Q_track_Zref_Z": "Q(Z)",
    "Q_track_Xref_X_fsnn": "Q(X)",
    "ctrl_total_latency_ms": "total latency",
    "ctrl_compute_latency_mean_ms": "compute latency",
}


def _plot_group(ax, df, cond_param, metrics, title, ylabel, show_title, show_legend):
    available = [m for m in metrics if m in df.columns and m != cond_param]
    if not available:
        ax.set_visible(False)
        return

    data = df[[cond_param] + available].copy()
    data = data.melt(
        id_vars=cond_param,
        value_vars=available,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    if data.empty:
        ax.set_visible(False)
        return

    cond_is_numeric = ptypes.is_numeric_dtype(data[cond_param])
    if ptypes.is_bool_dtype(data[cond_param]):
        data[cond_param] = data[cond_param].map({True: "True", False: "False"})
        cond_is_numeric = False
    elif not cond_is_numeric:
        data[cond_param] = data[cond_param].astype(str)

    order = sorted(data[cond_param].unique())

    mean_marker = "D"
    mean_marker_size = 5
    point_alpha = 0.1
    point_size = 3
    if cond_is_numeric and len(order) > 1 and not cond_param.startswith("R2"):
        sns.scatterplot(
            data=data,
            x=cond_param,
            y="value",
            hue="metric",
            alpha=point_alpha,
            size=point_size,
            ax=ax,
            legend=False,
        )
        sns.lineplot(
            data=data,
            x=cond_param,
            y="value",
            hue="metric",
            **psty.sns_errorbar_kwargs,
            marker=mean_marker,
            markersize=mean_marker_size,
            ax=ax,
            legend=show_legend,
            zorder=5,
        )
    elif cond_is_numeric and cond_param.startswith("R2"):
        sns.scatterplot(
            data=data,
            x=cond_param,
            y="value",
            hue="metric",
            alpha=point_alpha,
            size=point_size,
            ax=ax,
            legend=show_legend,
        )
    else:
        multiple_metrics = data["metric"].unique().size > 1
        dodge = 0.3 if multiple_metrics else False
        sns.stripplot(
            data=data,
            x=cond_param,
            y="value",
            hue="metric",
            dodge=dodge,
            alpha=point_alpha,
            size=point_size,
            ax=ax,
            order=order,
            legend=False,
        )
        sns.pointplot(
            data=data,
            x=cond_param,
            y="value",
            hue="metric",
            dodge=dodge,
            marker=mean_marker,
            markersize=mean_marker_size,
            **psty.sns_errorbar_kwargs,
            order=order,
            ax=ax,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=5,
        )
        if not show_legend and ax.get_legend():
            ax.get_legend().remove()

    if show_legend and ax.get_legend():
        handles, labels = ax.get_legend_handles_labels()
        labels = [metric_labels.get(label, label) for label in labels]
        ax.legend(handles, labels, title="", fontsize="small")

    if show_title:
        ax.set_title(title)
    ax.set_xlabel(cond_param)
    ax.set_ylabel(ylabel)
    if ylabel.lower().startswith("r2") or ylabel.lower().startswith("q"):
        ax.axhline(0, color="lightgray", lw=0.8, ls="--")


def main(cfg, results_csvs, outfiles):
    psty.style_plots()

    df = pd.concat([pd.read_csv(f) for f in results_csvs], ignore_index=True)
    if "alpha_stim" not in df.columns and {"k", "m"}.issubset(df.columns):
        df["alpha_stim"] = df["k"] / (df["m"] + df["k"])
    df["m"] = df["m"].fillna(0)
    df["signals"] = df["signals"].fillna("N/A")
    df["seed"] = df["seed"].astype("category")

    cond_params = []
    for col in df.columns:
        if col.startswith("R2_") or col.startswith("Q_"):
            break
        cond_params.append(col)
    cond_params.append("n_factors")
    cond_params.append("R2_fsnn_target_factors_mean")
    cond_params.append("R2_model_fit_X_Xhat_fsnn")

    n_rows = len(cond_params)
    n_cols = len(metric_groups)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.2 * n_cols, 2.2 * n_rows),
        layout="compressed",
        squeeze=False,
    )

    for i, cond_param in enumerate(cond_params):
        print(f"plotting {cond_param}")
        for j, (group_name, metrics, ylabel) in enumerate(metric_groups):
            ax = axs[i, j]
            show_title = i == 0
            show_legend = i % 3 == 0
            _plot_group(
                ax,
                df,
                cond_param,
                metrics,
                group_name,
                ylabel,
                show_title,
                show_legend,
            )

    for outfile in outfiles:
        fig.savefig(outfile)
        print(f"Saved {outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config.get("debug", False))

    main(snakemake.config, snakemake.input.csvs, snakemake.output)


if __name__ == "__main__" and "snakemake" not in locals():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-csvs",
        nargs="+",
        required=True,
        help="CSV files with results to plot",
    )
    parser.add_argument(
        "--outfiles",
        nargs="+",
        required=True,
        help="Output figure files (one or more of png, svg, pdf)",
    )
    args = parser.parse_args()

    main({}, args.results_csvs, args.outfiles)
